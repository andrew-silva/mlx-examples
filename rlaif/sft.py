# Copyright © 2023 Apple Inc.

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import utils
from mlx.utils import tree_flatten
from models.prompt_tuning import PromptTuning
from models.lora import LoRALinear


"""
python sft.py --prompt-tuning --save-file prompt_weights.npz --data-base increasing_mult_2_ --model /Users/andrewsilva/Desktop/research/code/tinkerings/mlx-examples/lora/tiny_llama --train
python sft.py --save-file lora_weights.npz --data-base increasing_mult_2_ --model /Users/andrewsilva/Desktop/research/code/tinkerings/mlx-examples/lora/tiny_llama --train
"""

def build_parser():
    parser = argparse.ArgumentParser(description="Soft Prompt finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    # Generation args
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="The maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="The prompt for generation",
        default=None,
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )
    parser.add_argument(
        "--prompt-tuning",
        action="store_true",
        help="Should we train with prompt tuning? If not, use LoRA",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory with {train, valid, test}.jsonl files",
    )
    parser.add_argument(
        "--data-base",
        type=str,
        default="",
        help="Base name for the .jsonl files. E.g., 'increasing_mult_2_'",
    )
    parser.add_argument(
        "--num-prompt-tokens",
        type=int,
        default=10,
        help="Number of prompt tokens to pre-pend",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Minibatch size.")
    parser.add_argument(
        "--iters", type=int, default=1000, help="Iterations to train for."
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=25,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-6, help="Adam learning rate."
    )
    parser.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=200,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-file",
        type=str,
        default=None,
        help="Load path to resume training with the given PEFT weights.",
    )
    parser.add_argument(
        "--save-file",
        type=str,
        default="peft_weights.npz",
        help="Save/load path for the trained PEFT weights.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return parser


class TuningDataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    """

    def __init__(self, path: Path, key: str = "text"):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        return len(self._data)


def load(train_args):
    ds_base = train_args.data_base
    ds_names = (f"{ds_base}train", f"{ds_base}valid", f"{ds_base}test")
    
    train_data, valid, test = (TuningDataset(Path(train_args.data) / f"{n}.jsonl") for n in ds_names)
    if train_args.train and len(train_data) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if train_args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if train_args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train_data, valid, test


def loss(mdl, inputs, targets, lengths):
    # Run model on inputs
    logits, _, _ = mdl(inputs)
    logits = logits.astype(mx.float32)

    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def iterate_batches(dset, tok, batch_size, train_mode=False):
    # Shuffle indices
    while True:
        indices = np.arange(len(dset))
        if train_mode:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = [tok.encode(dset[indices[i + j]]) for j in range(batch_size)]
            lengths = [len(x) for x in batch]

            # Check if any sequence is longer than 2048 tokens
            if max(lengths) > 2048:
                print(
                    "[WARNING] Some sequences are longer than 2048 tokens. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the max length
            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)

            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]
            batch = mx.array(batch_arr)
            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train_mode:
            break


def evaluate(mdl, dataset, loss_fn, tok, batch_size, num_batches):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(dataset, tok, batch_size),
    ):
        losses, toks = loss_fn(mdl, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens


def train(mdl, train_ds, val_set, optimizer, loss_fn, tok, train_args):
    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(mdl, loss_fn)

    losses = []
    val_losses = []
    n_tokens = 0

    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(train_args.iters),
        iterate_batches(train_ds, tok, train_args.batch_size, train_mode=True),
    ):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(mdl, *batch)

        # Model update
        optimizer.update(mdl, grad)
        mx.eval(mdl.parameters(), optimizer.state, lvalue)

        # Record loss
        losses.append(lvalue.item())
        n_tokens += toks.item()

        # Report training loss if needed
        if (it + 1) % train_args.steps_per_report == 0:
            train_loss = np.mean(losses[-train_args.steps_per_report:])

            stop = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {train_args.steps_per_report / (stop - start):.3f}, "
                f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
            )
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if it == 0 or (it + 1) % train_args.steps_per_eval == 0:
            stop = time.perf_counter()
            val_loss = evaluate(
                mdl, val_set, loss_fn, tok, train_args.batch_size, train_args.val_batches
            )
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )
            val_losses.append(val_loss)

            start = time.perf_counter()

        # Save prompt weights if needed
        if (it + 1) % train_args.save_every == 0:
            mx.savez(
                train_args.save_file, **dict(tree_flatten(mdl.trainable_parameters()))
            )
            print(f"Iter {it + 1}: Saved PEFT weights to {train_args.save_file}.")
    fn = ''
    if train_args.prompt_tuning:
        fn += 'prompt_tuning_'
    else:
        fn += 'lora_'
    plt.plot(losses)
    plt.savefig(f'{fn}train_losses.png')
    plt.plot(val_losses)
    plt.savefig(f'{fn}val_losses.png')


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Loading pretrained model")
    model, tokenizer, _ = utils.load(args.model)

    # Freeze all layers other than soft prompts
    model.freeze()
    if args.prompt_tuning:
        model = PromptTuning(num_tokens=args.num_prompt_tokens, model=model)
    else:
        for l in model.model.layers[len(model.model.layers) - args.lora_layers:]:
            l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
            l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
            if hasattr(l, "block_sparse_moe"):
                l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")

    print("Loading datasets")
    train_set, valid_set, test_set = load(args)

    # Resume training the given prompts.
    if args.resume_file is not None:
        print(f"Loading pretrained prompts from {args.resume_file}")
        model.load_weights(args.resume_file, strict=False)

    if args.train:
        print("Training")
        opt = optim.Adam(learning_rate=args.learning_rate)

        # Train model
        train(model, train_set, valid_set, opt, loss, tokenizer, args)

        # Save prompt weights
        mx.savez(args.save_file, **dict(tree_flatten(model.trainable_parameters())))

    # Load the soft prompt weights which we assume should exist by this point
    if not Path(args.save_file).is_file():
        raise ValueError(
            f"Prompt file {args.save_file} missing. "
            "Use --train to learn and save the prompts.npz."
        )
    model.load_weights(args.save_file, strict=False)

    if args.test:
        print("Testing")
        model.eval()
        test_loss = evaluate(
            model,
            test_set,
            loss,
            tokenizer,
            args.batch_size,
            num_batches=args.test_batches,
        )
        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    if args.prompt is not None:
        print("Generating")
        utils.generate(model, args.prompt, tokenizer, args)

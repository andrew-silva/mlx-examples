# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python ppo.py --log_with=wandb
"""
from dataclasses import dataclass, field
from typing import Optional

import mlx
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import load_dataset
# from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from mlx.utils import tree_flatten
from custom_ppo_trainer import PPOTrainer
from digit_seq_rewards import RewardFunction
from utils import set_seed, load
from config import PPOConfig
from models.prompt_tuning import PromptTuning
from models.lora import LoRALinear

tqdm.pandas()


@dataclass
class ScriptArguments:
    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})


parser = HfArgumentParser((ScriptArguments, PPOConfig))
args, ppo_config = parser.parse_args_into_dataclasses()
# TODO: Why does adaptive_kl break things? Over/underflow?
ppo_config.adap_kl_ctrl = False

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(ppo_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
if not args.use_peft:
    ref_model, _, _ = load('/Users/andrewsilva/Desktop/research/code/tinkerings/mlx-examples/lora/mlx_model')
    device_map = None
    peft_config = None
else:
    # peft_config = LoraConfig(
    #     r=args.lora_r,
    #     lora_alpha=args.lora_alpha,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    ref_model = None

model, tokenizer, _ = load('/Users/andrewsilva/Desktop/research/code/tinkerings/mlx-examples/lora/mlx_model')
model.freeze()
prompt_tuning = True
lora = False
if prompt_tuning:
    model = PromptTuning(num_tokens=10, model=model)
elif lora:
    lora_layers = 8
    for l in model.model.layers[len(model.model.layers) - lora_layers :]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

# for l in model.model.layers[:1]:
#     l.unfreeze()

p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
print(f"Total parameters {p:.3f}M")
p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
print(f"Trainable parameters {p:.3f}M")

# We retrieve the dataloader by calling the `build_dataset` function.
# dataset = build_dataset(tokenizer, ppo_config.query_dataset)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, data_collator=collator)

reward_function = RewardFunction(is_increasing=True, multiple_of=2)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

for epoch in range(150):
    text_in = 'Count up even numbers 2 8 20'
    batch = {
        'query': [text_in],
        # 'input_ids': mx.array(tokenizer.encode(text_in))
    }
    query_tensors = mx.array(tokenizer.encode(text_in))  # batch["input_ids"]

    # Get response from gpt2
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )

    batch["response"] = tokenizer.batch_decode(np.array(response_tensors))
    batch["ref_response"] = tokenizer.batch_decode(np.array(ref_response_tensors))
    response_tensors = [response_tensors[0]]
    ref_response_tensors = [ref_response_tensors[0]]

    # Compute sentiment score
    # texts = [q + r for q, r in zip(batch["query"], batch["response"])]

    scores = reward_function(batch['response'])  # Should we omit query in the scoring?
    rewards = [mx.array(scores)]

    ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
    ref_scores = reward_function(batch['ref_response'])
    batch["ref_rewards"] = ref_scores

    # Run PPO step
    stats = ppo_trainer.step([query_tensors], response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

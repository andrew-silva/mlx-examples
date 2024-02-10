# Fine-Tuning with Soft Prompts

This is an example of using MLX to fine-tune an LLM with soft prompts for a target task.[^prompt-tuning]. The example works with Llama and Mistral style
models available on Hugging Face.

In this example we'll use the WikiSQL[^wikisql] dataset to train the LLM to
generate SQL queries from natural language. However, the example is intended to
be general should you wish to use a custom dataset.

## Contents

* [Setup](#Setup)
  * [Convert](#convert)
* [Run](#Run)
  * [Fine-tune](#Fine-tune)
  * [Evaluate](#Evaluate)
  * [Generate](#Generate)
* [Results](#Results)
* [Fuse and Upload](#Fuse-and-Upload)
* [Custom Data](#Custom-Data)
* [Memory Issues](#Memory-Issues)


## Setup 

Install the dependencies:

```
pip install -r requirements.txt
```

### Convert

This step is optional if you want to quantize or change the default
data type of a pre-existing model.

You convert models using the `convert.py` script. This script takes a Hugging
Face repo as input and outputs a model directory (which you can optionally also
upload to Hugging Face).

To make a 4-bit quantized model, run:

```
python convert.py --hf-path <hf_repo> -q
```

For example, the following will make a 4-bit quantized Mistral 7B and by default
store it in `mlx_model`:

```
python convert.py --hf-path mistralai/Mistral-7B-v0.1 -q
```

For more options run:

```
python convert.py --help
```

You can upload new models to the [Hugging Face MLX
Community](https://huggingface.co/mlx-community) by specifying `--upload-name`
to `convert.py`.

## Run

The main script is `prompt_tuning.py`. To see a full list of options run:

```
python prompt_tuning.py --help
```

Note, in the following the `--model` argument can be any compatible Hugging
Face repo or a local path to a converted mdoel. 

### Fine-tune

To fine-tune a model use:

```
python prompt_tuning.py --model <path_to_model> \
               --train \
               --iters 600
```


By default, the adapter weights are saved in `prompts.npz`. You can specify
the output location with `--prompt-file`.

You can resume fine-tuning with an existing set of soft-prompts with `--resume-prompt-file
<path_to_prompts.npz>`. 

### Evaluate

To compute test set perplexity use:

```
python prompt_tuning.py --model <path_to_model> \
                        --prompt-file <path_to_prompts.npz> \
                        --test
```

### Generate

For generation use:

```
python prompt_tuning.py --model <path_to_model> \
                        --prompt-file <path_to_prompts.npz> \
                        --max-tokens 50 \
                        --prompt "table: 1-10015132-16
columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
Q: What is terrence ross' nationality
A: "
```

## Results

The initial validation loss for Llama 7B on the WikiSQL is 2.66 and the final
validation loss after 1000 iterations is 1.23. The table below shows the
training and validation loss at a few points over the course of training.

| Iteration | Train Loss | Validation Loss |
| --------- | ---------- | --------------- |
| 1         |    N/A     |      2.659      |
| 200       |    1.264   |      1.405      |
| 400       |    1.201   |      1.303      |
| 600       |    1.123   |      1.274      |
| 800       |    1.017   |      1.255      |
| 1000      |    1.070   |      1.230      |

The model trains at around 475 tokens per second on an M2 Ultra.

## Custom Data

You can make your own dataset for fine-tuning with soft prompts. You can specify the
dataset with `--data=<my_data_directory>`. Check the subdirectory `data/` to
see the expected format.

For fine-tuning (`--train`), the data loader expects a `train.jsonl` and a
`valid.jsonl` to be in the data directory. For evaluation (`--test`), the data
loader expects a `test.jsonl` in the data directory. Each line in the `*.jsonl`
file should look like:

```
{"text": "This is an example for the model."}
```

Note other keys will be ignored by the loader.

## Memory Issues

Fine-tuning a large model with soft-prompts requires a machine with a decent amount
of memory. Here are some tips to reduce memory use should you need to do so:

1. Try quantization. You can generate a quantized model
   with `convert.py` and the `-q` flag. See the [Setup](#setup) section for
   more details. 

2. Try using a smaller batch size with `--batch-size`. The default is `4` so
   setting this to `2` or `1` will reduce memory consumption. This may slow
   things down a little, but will also reduce the memory use.

3. Reduce the number of soft prompt tokens to fine-tune with `--num-prompt-tokens`. The default
   is `10`, so you can try `8` or `4`. This reduces the amount of memory
   needed for back propagation. It may also reduce the quality of the
   fine-tuned model if you are fine-tuning with a lot of data.

4. Longer examples require more memory. If it makes sense for your data, one thing
   you can do is break your examples into smaller
   sequences when making the `{train, valid, test}.jsonl` files.

For example, for a machine with 32 GB the following should run reasonably fast:

```
python prompt_tuning.py \
   --model mistralai/Mistral-7B-v0.1 \
   --train \
   --batch-size 1 \
   --num-prompt-tokens 10
```

The above command on an M1 Max with 32 GB runs at about 250 tokens-per-second.


[^prompt_tuning]: Refer to the [ACL paper](https://aclanthology.org/2021.emnlp-main.243/) for more details on soft prompts.
[^wikisql]: Refer to the [GitHub repo](https://github.com/salesforce/WikiSQL/tree/master) for more information about WikiSQL.

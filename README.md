# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Dataset

### Train BPE
``` sh
uv run cs336_basics/bpe.py --input_path=data/sync/raw/TinyStoriesV2-GPT4-train.txt --vocab_size=10000
uv run cs336_basics/bpe.py --input_path=data/sync/raw/owt_train.txt --vocab_size=32000 --pre_tokens_path=data/output/owt_train-pre_tokens.pkl
```

### Tokenization

``` sh
uv run cs336_basics/tokenizer/preprocess.py --data_path=data/sync/raw/TinyStoriesV2-GPT4-train.10k.txt \
    --pretrained_filepath=data/sync/bpe/TinyStoriesV2-GPT4-train-bpe.pkl \
    --output_path=data/sync/tokenized/TinyStoriesV2-GPT4-10k/train.npy

uv run cs336_basics/tokenizer/preprocess.py --data_path=datdata/sync/raw/owt_train.txt \
    --pretrained_filepath=data/sync/bpe/owt_train-bpe.pkl \
    --output_path=data/sync/tokenized/owt/train.npy
```

### Sync data

To upload the data to S3 bucket
``` sh
aws s3 sync ./data/sync s3://yanlinyc/cs336/assignment1-basics --profile yanlinyc
```

To download the data from S3 bucket
``` sh
aws s3 sync s3://yanlinyc/cs336/assignment1-basics ./data/sync --profile yanlinyc
```

## Train Model
``` sh
uv run cs336_basics/train.py --train_dataset_path=data/sync/tokenized/TinyStoriesV2-GPT4/train.npy
```

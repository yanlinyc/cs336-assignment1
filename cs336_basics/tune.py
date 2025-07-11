import os
from dataclasses import asdict

import numpy as np
import numpy.typing as npt

from cs336_basics.modules import TransformerLM
from cs336_basics.optim import AdamW
from cs336_basics.train_loop import (
    ModelConfig,
    OptimizerConfig,
    TrainingArguments,
    parse_arguments,
    train_loop,
)
from cs336_basics.utils import load_from_pretrained


def run_training(
    train_dataset: npt.NDArray[np.int64],
    training_args: TrainingArguments,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    eval_dataset: npt.NDArray[np.int64] | None = None,
    pretrained_checkpoint: str | os.PathLike | None = None,
):
    if pretrained_checkpoint:
        model, optimizer, iteration = load_from_pretrained(pretrained_checkpoint)
        print(
            f"Loaded pretrained model and optimizer from {pretrained_checkpoint} at iteration {iteration}."
        )
    else:
        iteration = 0
        print("No pretrained checkpoint provided, starting training from scratch.")
        model = TransformerLM(**asdict(model_config), device=training_args.device)
        optimizer = AdamW.from_pretrained(model, asdict(optimizer_config))

    train_loop(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        optimizer=optimizer,
        args=training_args,
        start_iteration=iteration,
    )


def main():
    config = parse_arguments()

    train_dataset = np.load(config.train_dataset_path, mmap_mode="r")
    eval_dataset = None
    if config.eval_dataset_path:
        eval_dataset = np.load(config.eval_dataset_path, mmap_mode="r")

    print(f"Loaded training dataset from {config.train_dataset_path}")
    if eval_dataset is not None:
        print(f"Loaded evaluation dataset from {config.eval_dataset_path}")

    run_training(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=config.training,
        model_config=config.model,
        optimizer_config=config.optimizer,
        pretrained_checkpoint=config.pretrained_checkpoint,
    )


if __name__ == "__main__":
    main()

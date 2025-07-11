from dataclasses import asdict

import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from cs336_basics.modules import TransformerLM
from cs336_basics.optim import AdamW
from cs336_basics.train_loop import (
    Config,
    from_dict,
    parse_arguments,
    train_loop,
)


def train_lm(tune_config: dict):
    config = from_dict(Config, tune_config["train_loop_config"])
    params_config = tune_config["params_config"]

    train_dataset = np.load(config.train_dataset_path, mmap_mode="r")
    eval_dataset = None
    if config.eval_dataset_path:
        eval_dataset = np.load(config.eval_dataset_path, mmap_mode="r")

    print(f"Loaded training dataset from {config.train_dataset_path}")
    if eval_dataset is not None:
        print(f"Loaded evaluation dataset from {config.eval_dataset_path}")

    iteration = 0
    config.optimizer.lr = params_config["lr"]
    print(f"Using learning rate: {params_config['lr']}")
    model = TransformerLM(**asdict(config.model), device=config.training.device)
    print(f"config.optimizer: {asdict(config.optimizer)}")
    optimizer = AdamW.from_pretrained(model, asdict(config.optimizer))

    # param_counts = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {param_counts:,}")

    train_loop(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        optimizer=optimizer,
        args=config.training,
        start_iteration=iteration,
        tune_params_config=params_config,
    )


def main():
    config = parse_arguments()

    assert config.tuning.enabled, "Tuning arguments must be provided for tuning."

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_lm),
            resources={
                "cpu": config.tuning.num_cpus_per_trial,
                "gpu": config.tuning.gpus_per_trial,
            },
        ),
        tune_config=tune.TuneConfig(
            num_samples=config.tuning.num_tune_samples,
            scheduler=ASHAScheduler(
                metric="eval_loss",
                mode="min",
                time_attr="training_iteration",
                max_t=config.tuning.max_iterations,
            ),
        ),
        param_space={
            "train_loop_config": asdict(config),
            "params_config": {
                "lr": tune.loguniform(1e-4, 1e-2),
            },
        },
    )

    results = tuner.fit()

    best_result = results.get_best_result("eval_loss", "min")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['eval_loss']}")


if __name__ == "__main__":
    main()

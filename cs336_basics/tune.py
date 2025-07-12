from dataclasses import asdict

import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from cs336_basics.modules import TransformerLM
from cs336_basics.optim import AdamW
from cs336_basics.train_loop import (
    Config,
    from_dict,
    parse_arguments,
    train_loop,
)


def train_lm(params_config: dict, train_loop_config: dict):
    config = from_dict(Config, train_loop_config)

    train_dataset = np.load(config.train_dataset_path, mmap_mode="r")
    eval_dataset = None
    if config.eval_dataset_path:
        eval_dataset = np.load(config.eval_dataset_path, mmap_mode="r")

    print(f"Loaded training dataset from {config.train_dataset_path}")
    if eval_dataset is not None:
        print(f"Loaded evaluation dataset from {config.eval_dataset_path}")

    iteration = 0
    config.optimizer.lr_scheduler_config.max_lr = params_config["lr"]
    print(f"Using learning rate: {params_config['lr']}")
    model = TransformerLM(**asdict(config.model), device=config.training.device)
    print(f"config.optimizer: {asdict(config.optimizer)}")
    optimizer = AdamW.from_pretrained(model, asdict(config.optimizer))

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

    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=config.tuning.max_concurrent_trials)
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_lm, train_loop_config=asdict(config)),
            resources={
                "cpu": config.tuning.num_cpus_per_trial,
                "gpu": config.tuning.gpus_per_trial,
            },
        ),
        tune_config=tune.TuneConfig(
            num_samples=config.tuning.num_tune_samples,
            # search_alg=algo,
            scheduler=ASHAScheduler(
                metric="eval_loss",
                mode="min",
                time_attr="training_iteration",
                max_t=config.tuning.max_iterations,
            ),
        ),
        param_space={
            "lr": tune.loguniform(1e-5, 1e-2),
        },
    )

    results = tuner.fit()

    best_result = results.get_best_result("eval_loss", "min")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['eval_loss']}")


if __name__ == "__main__":
    main()

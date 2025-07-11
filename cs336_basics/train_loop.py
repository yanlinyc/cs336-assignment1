import os
import tomllib
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TypeVar, get_type_hints

import numpy as np
import numpy.typing as npt
import torch
import tyro
from ray import tune
from ray.air.integrations.wandb import setup_wandb
from tqdm.auto import tqdm

import wandb
from cs336_basics.modules import TransformerLM, cross_entropy
from cs336_basics.optim import AdamW
from cs336_basics.utils import save_checkpoint
from cs336_basics.utils.data import get_batch

T = TypeVar("T")


def from_dict(cls: type[T], data: dict) -> T:
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")

    fieldtypes = get_type_hints(cls)
    init_args = {}
    for field_name, field_type in fieldtypes.items():
        if field_name in data:
            value = data[field_name]
            if is_dataclass(field_type) and isinstance(value, dict):
                init_args[field_name] = from_dict(field_type, value)
            else:
                init_args[field_name] = value
    return cls(**init_args)


@dataclass
class LRSchedulerConfig:
    cls: str = "CosineLRScheduler"
    warmup_iters: int = 0
    cosine_cycle_iters: int = 1000
    min_lr: float = 1e-5
    max_lr: float = 1e-3


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0.01
    lr_scheduler_config: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


@dataclass
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 1024
    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    rope_theta: float = 10000.0


@dataclass
class TrainingArguments:
    output_dir: str | os.PathLike = "output/checkpoints"
    context_length: int = 1024
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_iterations: int = 1000
    eval_num_batches: int = 100
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 100
    device: str = "cpu"
    debug_fixed_minibatch: bool = False
    random_seed: int | None = None


@dataclass
class TuningArguments:
    enabled: bool = False
    output_dir: str | os.PathLike = "output/tuning"
    num_tune_samples: int = 10
    gpus_per_trial: float = 0.5
    num_cpus_per_trial: int = 2
    max_iterations: int = 10


@dataclass
class Config:
    train_dataset_path: str | os.PathLike = ""
    eval_dataset_path: str | os.PathLike | None = None
    config: str = "training_config.toml"
    pretrained_checkpoint: str | os.PathLike | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingArguments = field(default_factory=TrainingArguments)
    tuning: TuningArguments = field(default_factory=TuningArguments)


def dict_to_dataclass(cls, data: dict):
    """Recursively populate a dataclass from a dictionary"""
    if not hasattr(cls, "__dataclass_fields__"):
        return data
    kwargs = {}
    for field_name, field_type in cls.__dataclass_fields__.items():
        if field_name in data:
            field_val = data[field_name]
            kwargs[field_name] = dict_to_dataclass(field_type.type, field_val)
    return cls(**kwargs)


def train_loop(
    train_dataset: npt.NDArray[np.int64],
    eval_dataset: npt.NDArray[np.int64] | None,
    model: TransformerLM,
    optimizer: AdamW,
    args: TrainingArguments,
    start_iteration: int = 0,
    tune_params_config: dict | None = None,
    verbose: bool = False,
):
    if verbose:
        print(f"Starting training with model: {model}")
        print(f"optimizer: {optimizer}")
        print(f"Training arguments: {args}")
        print(f"Tuning parameters: {tune_params_config}")

    if tune_params_config:
        wandb_run = setup_wandb(
            project="cs336", entity="yanlinyc-thatcher", config=tune_params_config
        )
    else:
        wandb_run = wandb.init(
            project="cs336",
            entity="yanlinyc-thatcher",
            name=f"train-{model.canonical_name}",
            config={
                "model_config": model.config,
                "optimizer_config": optimizer.config,
                "training_args": asdict(args),
            },
        )

    args.output_dir = os.path.join(args.output_dir, wandb.run.id)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.random_seed is not None:
        import random

        print(f"Setting random seed to {args.random_seed}")
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    fixed_starts = None
    if args.debug_fixed_minibatch:
        fixed_starts = np.random.randint(
            0, train_dataset.shape[0] - args.context_length, size=args.train_batch_size
        )
        print(f"Debug mode: Using fixed minibatch indices: {fixed_starts}")

    model.train(True)
    model.to(args.device)
    if start_iteration > 0:
        print(f"Resuming training from iteration {start_iteration}.")

    for i in tqdm(
        range(start_iteration, start_iteration + args.num_iterations),
        desc="Training",
        initial=start_iteration,
    ):
        step = i + 1
        inputs, targets = get_batch(
            train_dataset,
            batch_size=args.train_batch_size,
            context_length=args.context_length,
            device=args.device,
            fixed_starts=fixed_starts,
        )

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = cross_entropy(predictions, targets)
        loss.backward()

        optimizer.step()

        if args.logging_steps > 0 and step % args.logging_steps == 0:
            if verbose:
                print(
                    f"Step {step}/{args.num_iterations}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
                )
            wandb_run.log(
                {
                    "step": step,
                    "loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                | compute_norms(model)
            )

        if eval_dataset is not None and args.eval_steps > 0 and step % args.eval_steps == 0:
            running_vloss = 0.0
            model.eval()
            with torch.no_grad():
                for _ in range(args.eval_num_batches):
                    eval_inputs, eval_targets = get_batch(
                        eval_dataset,
                        batch_size=args.eval_batch_size,
                        context_length=args.context_length,
                        device=args.device,
                    )
                    eval_predictions = model(eval_inputs)
                    eval_loss = cross_entropy(eval_predictions, eval_targets)
                    running_vloss += eval_loss.item()
                avg_vloss = running_vloss / args.eval_num_batches
            model.train(True)
            if verbose:
                print(
                    f"Step {step}/{args.num_iterations}, Loss: {loss.item():.4f}, Eval Loss: {avg_vloss:.4f}"
                )
            eval_report = {"eval_loss": avg_vloss}
            wandb_run.log(eval_report)
            if tune_params_config:
                tune.report(eval_report)

        if args.save_steps > 0 and step % args.save_steps == 0:
            output_path = os.path.join(args.output_dir, f"checkpoint-{step}.pt")
            # TODO: figure if we should use step or i
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=output_path,
            )
            if verbose:
                print(f"Saved checkpoint at step {step} to {output_path}")
    output_path = os.path.join(args.output_dir, f"final_checkpoint-{args.num_iterations}.pt")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=args.num_iterations,
        out=output_path,
    )
    print(f"Saved final checkpoint to {output_path}")
    wandb_run.finish()


def parse_arguments() -> Config:
    # First parse command-line args to get `--config` if present
    base_config = tyro.cli(Config, exit_on_help=False)

    # Load TOML config if provided
    if base_config.config:
        with open(base_config.config, "rb") as f:
            toml_data = tomllib.load(f)
        config_from_file = dict_to_dataclass(Config, {"config": base_config.config, **toml_data})
    else:
        config_from_file = Config()

    # Final parse: use TOML as defaults, allow CLI to override
    final_config = tyro.cli(Config, default=config_from_file)

    if not final_config.train_dataset_path:
        raise ValueError("train_dataset_path must be specified.")

    print(f"Final configuration: {final_config}")
    return final_config


def compute_norms(model: TransformerLM) -> dict[str, float]:
    norms = {
        "weights_norm": 0.0,
        "gradients_norm": 0.0,
    }

    # Global weight norm
    weight_sq_sum = 0.0
    for p in model.parameters():
        if p.requires_grad:
            weight_sq_sum += p.data.pow(2).sum()
    norms["weights_norm"] = torch.sqrt(weight_sq_sum).item()

    # Global gradient norm
    grad_sq_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sq_sum += p.grad.detach().pow(2).sum()
    norms["gradients_norm"] = torch.sqrt(grad_sq_sum).item()

    return norms

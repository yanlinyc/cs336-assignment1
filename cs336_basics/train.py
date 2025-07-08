import os
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from tqdm.auto import tqdm

import wandb
from cs336_basics.modules import TransformerLM, cross_entropy
from cs336_basics.optim import AdamW
from cs336_basics.utils import save_checkpoint
from cs336_basics.utils.data import get_batch


@dataclass
class TrainingArguments:
    output_dir: str | os.PathLike
    context_length: int = 1024
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_iterations: int = 1000
    eval_num_batches: int = 100
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 100
    device: str = "cpu"


def train_loop(
    train_dataset: npt.NDArray[np.int64],
    eval_dataset: npt.NDArray[np.int64],
    model: TransformerLM,
    optimizer: AdamW,
    args: TrainingArguments,
):
    wandb.init(
        project="cs336",
        entity="yanlinyc-thatcher",
        name=f"train-{model.canonical_name}",
        config=(
            model.config
            | {
                "train_batch_size": args.train_batch_size,
                "num_iterations": args.num_iterations,
            }
        ),
    )
    model.train(True)
    model.to(args.device)
    for i in tqdm(range(args.num_iterations), desc="Training"):
        step = i + 1
        inputs, targets = get_batch(
            train_dataset,
            batch_size=args.train_batch_size,
            context_length=args.context_length,
            device=args.device,
        )

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = cross_entropy(predictions, targets)
        loss.backward()

        optimizer.step()

        if step % args.logging_steps == 0:
            print(
                f"Step {step}/{args.num_iterations}, "
                f"Loss: {loss.item():.4f}, "
                f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
            )
            wandb.log(
                {
                    "step": step,
                    "loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        if step % args.eval_steps == 0:
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
            print(
                f"Step {step}/{args.num_iterations}, Loss: {loss.item():.4f}, Eval Loss: {avg_vloss:.4f}"
            )
            wandb.log({"eval_loss": avg_vloss})

        if step % args.save_steps == 0:
            output_path = os.path.join(args.output_dir, f"checkpoint-{step}.pt")
            # TODO: figure if we should use step or i
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                path=output_path,
            )
            print(f"Saved checkpoint at step {step} to {output_path}")
    print("Training complete.")
    output_path = os.path.join(args.output_dir, f"final_checkpoint-{args.num_iterations}.pt")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=args.num_iterations,
        path=output_path,
    )
    print(f"Saved final checkpoint to {output_path}")
    wandb.finish()


def run_training(
    train_dataset: npt.NDArray[np.int64],
    eval_dataset: npt.NDArray[np.int64],
    args: TrainingArguments,
    pretrained_checkpoint: str | os.PathLike | None = None,
):
    model = TransformerLM()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loop(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        optimizer=optimizer,
        args=args,
    )

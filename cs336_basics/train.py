import os

import numpy as np
import numpy.typing as npt
import torch
from tqdm.auto import tqdm

import wandb
from cs336_basics.modules import TransformerLM, cross_entropy
from cs336_basics.optim import AdamW
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.utils import save_checkpoint
from cs336_basics.utils.data import get_batch


def train_loop(
    train_dataset: npt.NDArray[np.int64],
    eval_dataset: npt.NDArray[np.int64],
    tokenizer: BPETokenizer,
    model: TransformerLM,
    optimizer: AdamW,
    output_dir: str | os.PathLike,
    context_length: int,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_iterations: int = 1000,
    eval_num_batches: int = 100,
    save_steps: int = 100,
    eval_steps: int = 100,
    logging_steps: int = 100,
    device: str = "cpu",
):
    wandb.init(
        project="cs336",
        entity="yanlinyc-thatcher",
        name="train-llm",
        config={
            "vocab_size": model.vocab_size,
            "context_length": context_length,
            "num_layers": model.num_layers,
            "d_model": model.d_model,
            "num_heads": model.num_heads,
            "d_ff": model.d_ff,
            "rope_theta": model.rope.theta,
            "batch_size": train_batch_size,
            "num_iterations": num_iterations,
        },
    )
    model.train(True)
    model.to(device)
    for i in tqdm(range(num_iterations), desc="Training"):
        step = i + 1
        inputs, targets = get_batch(
            train_dataset,
            batch_size=train_batch_size,
            context_length=context_length,
            device=device,
        )

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = cross_entropy(predictions, targets)
        loss.backward()

        optimizer.step()

        if step % logging_steps == 0:
            print(
                f"Step {step}/{num_iterations}, "
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

        if step % eval_steps == 0:
            running_vloss = 0.0
            model.eval()
            with torch.no_grad():
                for _ in range(eval_num_batches):
                    eval_inputs, eval_targets = get_batch(
                        eval_dataset,
                        batch_size=eval_batch_size,
                        context_length=context_length,
                        device=device,
                    )
                    eval_predictions = model(eval_inputs)
                    eval_loss = cross_entropy(eval_predictions, eval_targets)
                    running_vloss += eval_loss.item()
                avg_vloss = running_vloss / eval_num_batches
            model.train(True)
            print(
                f"Step {step}/{num_iterations}, Loss: {loss.item():.4f}, Eval Loss: {avg_vloss:.4f}"
            )
            wandb.log({"eval_loss": avg_vloss})

        if step % save_steps == 0:
            output_path = os.path.join(output_dir, f"checkpoint-{step}.pt")
            # TODO: figure if we should use step or i
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                path=output_path,
            )
            print(f"Saved checkpoint at step {step} to {output_path}")
    print("Training complete.")
    output_path = os.path.join(output_dir, f"final_checkpoint-{num_iterations}.pt")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=num_iterations,
        path=output_path,
    )
    print(f"Saved final checkpoint to {output_path}")
    wandb.finish()

import wandb

from ACIL.trainer.callbacks.callbacks import BaseCallback


class Wandb(BaseCallback):
    """Callback to log training information to Weights & Biases."""

    def __call__(self, loss, accuracy, epoch, lr, **_) -> None:
        """Log training information to Weights & Biases."""
        wandb.log({"Train/Loss": loss, "Train/Accuracy": accuracy, "Train/Epoch": epoch, "Train/LR": lr})

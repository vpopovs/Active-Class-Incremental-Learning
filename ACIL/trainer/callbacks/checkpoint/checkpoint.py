from ACIL.trainer.callbacks.callbacks import BaseCallback


class Checkpoint(BaseCallback):
    """Callback to save the model checkpoint."""

    def __call__(self, trainer, epoch, **_) -> None:
        """Saves the model checkpoint on an interval."""
        if epoch and epoch % self.cfg.every_n_epochs == 0:
            trainer.model.save(f"epoch_{epoch}")
            self.log.debug(f"Saved checkpoint at epoch {epoch}.")

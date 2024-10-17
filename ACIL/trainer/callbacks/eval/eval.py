from ACIL.trainer.callbacks.callbacks import BaseCallback


class Eval(BaseCallback):
    """Callback to evaluate the model."""

    def __call__(self, trainer, epoch, **_) -> None:
        """Evaluates the model on an interval."""
        if (epoch | self.cfg.at_start) and (epoch + 1) % self.cfg.every_n_epochs == 0:
            trainer.eval(loader=self.cfg.loader)

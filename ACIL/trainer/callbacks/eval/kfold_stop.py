from __future__ import annotations

from typing import TYPE_CHECKING

from ACIL.trainer.callbacks.eval.stop import EvalStop

if TYPE_CHECKING:
    from ACIL.trainer.trainer import BaseTrainer as Trainer


class KFoldStop(EvalStop):
    """Variant of EvalStop that stops training after n_folds without improvement."""

    def __post_init__(self):
        """Initialize the callback."""
        self.reset()
        self.patience = int(self.cfg.patience * self.cfg.n_folds)
        self.warmup = int(self.cfg.warmup * self.cfg.n_folds)
        self.log.info(f"Initalized {self.cfg.n_folds}-FoldStop with patience {self.patience} and warmup {self.warmup}.")

    def __call__(self, trainer: Trainer, epoch: int, **_) -> bool:
        """Forward call to parent class and reset fold counter if stopping training."""
        res = super().__call__(trainer, epoch)
        if res:
            trainer.model._n_fold = 0
        return res

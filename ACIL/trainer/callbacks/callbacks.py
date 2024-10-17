from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from hydra.utils import instantiate

from ACIL.utils.base import Base

if TYPE_CHECKING:
    from ACIL.trainer.trainer import BaseTrainer as Trainer


class Callbacks(Base):
    """
    Callback class which is called every training step.

    The initialised callbacks get the opportunity to use the provided information
        and are able to terminate the training loop.
    """

    def __init__(self, **cfg: dict) -> None:
        """
        Initializes the Callbacks object.

        Args:
            **cfg (dict): Remaining Hydra configurations.
        """
        super().__init__(cfg)
        self.callbacks = {name: instantiate(callback) for name, callback in self.cfg.items()}
        self.log.info(f"Loaded callbacks: {', '.join(self.callbacks.keys())}.")

    def __call__(
        self,
        trainer: Trainer,
        epoch: int,
        data: Tuple,
        outputs: Any,
        lr: Optional[float] = None,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
    ) -> bool:
        """
        Activates the callbacks passing trainer information.

        Args:
            trainer (Trainer): Trainer object.
            epoch (int): Current epoch.
            data (Tuple): Last batch of data tuple.
            outputs (Any) Last batch of outputs for data tuple.
            lr (Optional[float]): Current learning rate.
            loss (Optional[float]): Current loss.
            accuracy (Optional[float]): Current accuracy.
        Returns:
            bool: Whether to break the training loop.
        """
        return any(
            callback(trainer=trainer, epoch=epoch, data=data, outputs=outputs, lr=lr, loss=loss, accuracy=accuracy)
            for callback in self.callbacks.values()
        )


class BaseCallback(Base):
    """Abstract Callback class that handles the callback logic. All callbacks should inherit from this class."""

    def __init__(self, **cfg: dict):
        """Initializes the BaseCallback object."""
        super().__init__(cfg)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Post-initialization method that can be used by the child class."""

    def __call__(self, **kwargs) -> Union[bool, None]:
        """Abstract method that should be implemented by the child class.

        Args:
            kwargs (dict): Keyword arguments to be used by the callback:
                trainer (Trainer): Trainer object.
                epoch (int): Current epoch.
                data (Tuple): Last batch of data tuple.
                outputs (Any) Last batch of outputs for data tuple.
                lr (Optional[float]): Current learning rate.
                loss (Optional[float]): Current loss.
                accuracy (Optional[float]): Current accuracy.
        Returns:
            bool: Whether to break the training loop.
        """
        raise NotImplementedError

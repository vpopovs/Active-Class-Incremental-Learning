from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from hydra.utils import instantiate

from ACIL.utils.base import Base

if TYPE_CHECKING:
    from ACIL.data.data import Data
    from ACIL.trainer.trainer import BaseTrainer as Trainer


class BaseModel(torch.nn.Module, Base):
    """
    Abstract Model class that handles model creation and specific model steps.
    All models should inherit from this class and implement the necessary methods.
    """

    def __init__(
        self,
        device: torch.device,
        data: Data,
        **cfg: dict,
    ):
        """
        Initializes the BaseModel with the device and data, and sets up the model architecture.

        Args:
            device (torch.device): Device on which the model will be trained.
            data (Data): Data object.
            **cfg (dict): Remaining Hydra configurations.
        """

        super().__init__()
        Base.__init__(self, cfg)

        self.device = device
        self.data = data
        self.n_classes = data.n_classes

        self.network = self.init_network()
        self.criterion = self.init_criterion()
        self.log.debug(f"Loaded criterion:\n{self.criterion}")

    def init_network(self) -> torch.nn.Module:
        """
        Initializes the network architecture.

        Returns:
            torch.nn.Module: The network model.
        """
        network = instantiate(self.cfg.network, n_classes=self.n_classes, device=self.device)
        network.to(self.device)
        return network

    def init_criterion(self) -> torch.nn.Module:
        """
        Initializes the loss criterion, if defined in the configuration.

        Returns:
            torch.nn.Module: The loss criterion.
        """
        if not self.cfg.get("criterion"):
            self.log.warning("No criterion loaded. Assuming criterion is defined in network.")
            return None
        criterion = instantiate(self.cfg.criterion)
        criterion.to(self.device)
        return criterion

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overrides the call method to directly forward input through the network.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output from the forward pass of the network.
        """
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the network.
        """
        return self.criterion.post_forward(self.network(x))

    def step(self, trainer: Trainer, data: torch.Tensor, target: torch.Tensor) -> Any:
        """
        Perform a single training step, to be implemented by the inherited model.

        Args:
            trainer (Trainer): The trainer object managing the training process.
            data (torch.Tensor): Input data for the step.
            target (torch.Tensor): Target labels or values for the step.
        Returns:
            Any: The output of the model.
        """
        if self.network.training:
            with torch.set_grad_enabled(True):
                output = self.network(data)
                loss = self.criterion(output, target)
                loss.backward()
                output = self.criterion.post_forward(output)
                return loss.item(), output
        return None, self.network(data)

    def pretrain(self, *args: Any, **kwargs: Any) -> None:
        """
        Optional method to be called before training starts. Can be overridden in subclasses.
        """

    def posttrain(self, *args: Any, **kwargs: Any) -> None:
        """
        Optional method to be called after training ends. Can be overridden in subclasses.
        """

    def preepoch(self, *args: Any, **kwargs: Any) -> None:
        """
        Method called before each training epoch to perform any necessary preparations.
        This implementation clears the CUDA cache.
        """
        torch.cuda.empty_cache()

    def postepoch(self, *args: Any, **kwargs: Any) -> None:
        """
        Optional method called after each training epoch. Can be overridden in subclasses.
        """

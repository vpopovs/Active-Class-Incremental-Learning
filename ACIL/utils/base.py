from __future__ import annotations

import logging
import os

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


class Base:
    """Abstract Base class for all classes in the project, providing common functionalities."""

    def __init__(self, cfg: DictConfig):
        """
        Initializes the Base object.
        Creates `self.log`, `self.cfg` and ensures the `output_dir` exists.

        Args:
            cfg (DictConfig): Hydra configuration, saved as HydraConfig as `self.cfg`
        """
        self.name = self.__module__.split(".")
        if len(self.name) > 1:
            self.name = [self.name[1][0].capitalize(), self.name[-1]]
        self.log = logging.getLogger(":".join(self.name))
        self.cfg = cfg if isinstance(cfg, DictConfig) else DictConfig(cfg)
        if "output_dir" in cfg:
            self.output_dir_base = HydraConfig.get().runtime.output_dir
            self.output_dir = os.path.join(self.output_dir_base, cfg["output_dir"])
            os.makedirs(self.output_dir, exist_ok=True)

    def __str__(self):
        """Returns the name of the class."""
        return self.name

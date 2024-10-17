import os

import torch
import torchvision.models as models
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss

from ACIL.utils.base import Base
from external.moco.moco.moco.builder import MoCo as MoCo_model


class MoCo(Base):
    def __init__(self, device, data, **cfg: DictConfig):
        super().__init__(cfg)
        self.device = device
        self._init_modules()

    def _init_modules(self):
        arch = self.cfg.network.arch
        del self.cfg.network.arch
        if hasattr(self.cfg.network, "transforms"):
            del self.cfg.network.transforms

        self.network = torch.nn.parallel.DistributedDataParallel(
            MoCo_model(models.__dict__[arch], **self.cfg.network).cuda()
        ).to(self.device)
        self.log.info(f"Loaded model:\n{self.network}")

        self.criterion = CrossEntropyLoss()
        self.criterion.to(self.device)
        self.log.info(f"Loaded criterion:\n{self.criterion}")

    def __call__(self, x):
        return self.network(x)

    def step(self, im_q, im_k):
        if self.network.training:
            with torch.set_grad_enabled(True):
                output, target = self.network(im_q, im_k)
                loss = self.criterion(output, target)
                loss.backward()
                return loss.item(), output, target
        return None, self.network(im_q, im_k)

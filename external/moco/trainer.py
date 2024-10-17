from __future__ import annotations

import math
import os
import sys
from typing import TYPE_CHECKING

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchvision import transforms

from ACIL.main import Runner
from ACIL.utils.base import Base
from ACIL.utils.module import import_from_cfg
from ACIL.utils.subsets import Subset

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from ACIL.data.data import Data

REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moco")
sys.path.append(REPO_PATH)

from external.moco.moco.main_moco import accuracy, save_checkpoint
from external.moco.moco.moco.loader import TwoCropsTransform


def build_transforms(transform):
    transform_class, transform_attr = list(transform.items())[0]
    transform_attr = {} if transform_attr is None else transform_attr
    if not transform_class.startswith("torchvision"):
        return import_from_cfg(transform_class)(**transform_attr).get()
    else:
        if transform_class == "torchvision.transforms.RandomApply":
            nested_transform_attr = {}
            nested_transform_attr["transforms"] = [build_transforms(t) for t in transform_attr]
            return import_from_cfg(transform_class)(**nested_transform_attr)
        return import_from_cfg(transform_class)(**transform_attr)


class MoCo_Trainer(Base):
    def __init__(self, device, model, data: Data, **cfg: DictConfig):
        # setup transforms
        super().__init__(cfg)
        self.device = device
        self.transforms = TwoCropsTransform(
            transforms.Compose([build_transforms(transform) for transform in self.cfg.transforms])
        )
        self.log.info(f"Loaded MoCo transforms:\n{self.transforms}\n{self.transforms.base_transform}")
        self.train_data = Subset(data.train_plain_loader.dataset, transform=self.transforms)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.workers,
            pin_memory=True,
            drop_last=True,
        )

        self.optimizer = instantiate(self.cfg.optimizer, model.network.parameters())
        self.model = model

        if self.cfg.resume:
            if os.path.isfile(self.cfg.resume):
                print("=> loading checkpoint '{}'".format(self.cfg.resume))
                if self.cfg.gpu is None:
                    checkpoint = torch.load(self.cfg.resume)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = "cuda:{}".format(self.cfg.gpu)
                    checkpoint = torch.load(self.cfg.resume, map_location=loc)
                self.cfg.start_epoch = checkpoint["epoch"]
                self.model.network.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print("=> loaded checkpoint '{}' (epoch {})".format(self.cfg.resume, checkpoint["epoch"]))
            else:
                print("=> no checkpoint found at '{}'".format(self.cfg.resume))
        print(self.model)
        os.chdir(REPO_PATH)

    def run(self):
        for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
            self.log.info(f"Epoch {epoch + 1}/{self.cfg.epochs}")
            self.adjust_learning_rate(epoch)
            self.train(epoch)
            if epoch % self.cfg.save_every == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": self.cfg.arch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    is_best=False,
                    filename="checkpoint_{:04d}.pth.tar".format(epoch),
                )

    def adjust_learning_rate(self, epoch):
        """Decay the learning rate based on schedule"""
        lr = self.cfg.optimizer.lr
        if self.cfg.cos:  # cosine lr schedule
            lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / self.cfg.epochs))
        else:  # stepwise lr schedule
            for milestone in self.cfg.schedule:
                lr *= 0.1 if epoch >= milestone else 1.0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(self, epoch):
        self.model.network.train()
        total_loss = 0.0
        total_acc_1 = 0.0
        total_acc_5 = 0.0
        img_counter = 0
        for images in self.train_loader:
            img_counter += len(images[0])
            self.optimizer.zero_grad()
            im_q, im_k = [x.to(self.device) for x in images]
            loss, output, target = self.model.step(im_q, im_k)
            self.optimizer.step()

            total_loss += loss.item()
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            total_acc_1 += acc1[0]
            total_acc_5 += acc5[0]

        wandb.log(
            {
                "moco/loss": total_loss,
                "moco/acc_1": total_acc_1 / img_counter,
                "moco/acc_5": total_acc_5 / img_counter,
                "moco/epoch": epoch,
            }
        )


@hydra.main(version_base="1.2", config_path="../../src", config_name="main")
def main(cfg: DictConfig):
    runner = Runner(cfg)
    runner.build()
    runner.run()


if __name__ == "__main__":
    sys.path.append(REPO_PATH)
    raise NotImplementedError
    # Runner.build = build
    # Runner.run = run
    # main()

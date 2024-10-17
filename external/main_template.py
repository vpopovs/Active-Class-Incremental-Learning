from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import hydra
from hydra.utils import instantiate

from ACIL.main import Runner

if TYPE_CHECKING:
    from omegaconf import DictConfig

REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), NotImplementedError)


def build(self):
    data = instantiate(self.cfg.data, device=self.device, _recursive_=False)
    raise NotImplementedError


def run(self):
    raise NotImplementedError


@hydra.main(version_base="1.2", config_path="../../src", config_name="main")
def main(cfg: DictConfig):
    runner = Runner(cfg)
    runner.build()
    runner.run()


if __name__ == "__main__":
    sys.path.append(REPO_PATH)

    Runner.build = build
    Runner.run = run
    main()

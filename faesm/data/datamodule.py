from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch
from byprot import utils
from byprot.datamodules import register_datamodule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from faesm.data.dataset import load_mix_dataset, setup_dataloader
from datasets import load_dataset


log = utils.get_logger(__name__)

@register_datamodule('mixed')
class MixedDataModule(LightningDataModule):
    def __init__(
        self,
        max_tokens: int = 6000,
        max_len: int = 1022,
        num_workers: int = 0,
        pin_memory: bool = False,
        mini_run: bool = False,
        mix_probabilities: List[float] = [0.8, 0.2],
        seed: int = 42,
        streaming: bool = True,
        max_num_reps=None,
        max_rep_subseq_len=None,
        min_entropy_cutoff=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: `self.train_dataset`, `self.valid_dataset`, `self.test_dataset`.
        This method is called by Lightning when doing `trainer.fit()` and `trainer.test()`.
        """
        # For training
        if stage == "fit":
            # Load train split
            log.info(f"Loading mixed dataset for TRAIN with probabilities {self.hparams.mix_probabilities}.")
            self.train_dataset = load_mix_dataset(
                streaming=self.hparams.streaming,
                mix_probabilities=self.hparams.mix_probabilities,
                seed=self.hparams.seed,
            )
            # Load validation split
            log.info(f"Loading mixed dataset for VALID with probabilities {self.hparams.mix_probabilities}.")
            self.valid_dataset = load_dataset("zhangzhi/Uniref50", split="valid", streaming=self.hparams.streaming)

            # Optional: subsample if mini_run is True
            if self.hparams.mini_run:
                # Example approach: take a small number of items
                log.info("Applying mini_run subsampling (train/valid).")
                self.train_dataset = self.train_dataset.take(500)  # arbitrary small subset
                self.valid_dataset = self.valid_dataset.take(200)

        # For testing
        elif stage == "test" or stage == "predict":
            # Load test split
            log.info(f"Loading mixed dataset for TEST with probabilities {self.hparams.mix_probabilities}.")
            self.test_dataset = load_dataset("zhangzhi/Uniref50", split="test", streaming=self.hparams.streaming)
            if self.hparams.mini_run:
                self.test_dataset = self.test_dataset.take(300)
        else:
            raise ValueError(f"Invalid stage: {stage}.")

        self.stage = stage

    def train_dataloader(self):
        print(self.hparams.max_tokens)
        return setup_dataloader(
            mixed_dataset=self.train_dataset,
            max_tokens=self.hparams.max_tokens,
            max_len=self.hparams.max_len,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            min_entropy=self.hparams.min_entropy_cutoff,
        )

    def val_dataloader(self):
        return setup_dataloader(
            mixed_dataset=self.valid_dataset,
            max_tokens=self.hparams.max_tokens,
            max_len=self.hparams.max_len,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return setup_dataloader(
            mixed_dataset=self.test_dataset,
            max_tokens=self.hparams.max_tokens,
            max_len=self.hparams.max_len,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
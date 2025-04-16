from faesm.scripts.train_utils import get_optimizer, get_scheduler
import torch
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: DictConfig,
        lr_scheduler: DictConfig = None,
        mask_strategy: DictConfig = None,
        ):
        """
        Args:
            model (nn.Module): The language model that takes input_ids and returns logits.
                               It must have a 'tokenizer' attribute with 'mask_token_id' and optionally 'pad_token_id'.
            learning_rate (float): Learning rate for the optimizer.
            mask_prob (float): The probability with which tokens will be masked.
        """
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.model = model
        
    def _is_maskable(self, input_ids: torch.Tensor):
        return (
            (input_ids != self.tokenizer.pad_token_id) 
            & (input_ids != self.tokenizer.cls_token_id)
            & (input_ids != self.tokenizer.eos_token_id)
        )
    def noise_x0(self, input_ids: torch.Tensor):
        labels = input_ids.clone()
        B, L = input_ids.size()
        min_mask_ratio, max_mask_ratio = self.hparams.mask_strategy.get("min_mask_ratio", 0.15), self.hparams.mask_strategy.get("max_mask_ratio", 0.25)
        mask_ratio = torch.rand(B, 1) * (max_mask_ratio - min_mask_ratio) + min_mask_ratio
        mask = torch.rand_like(input_ids.float()) < mask_ratio
        mask &= self._is_maskable(input_ids)
        input_ids[mask] = self.tokenizer.mask_token_id
        return mask_ratio, input_ids, labels

    def forward(self, input_ids: torch.Tensor):
        """
        Forward pass through the model.
        Assumes that self.model returns logits given input_ids.
        """
        return self.model(input_ids)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, loss_weight: torch.Tensor=None):
        """
        Compute the cross entropy loss per sample.
        First, compute the per-token loss (with no reduction), then reduce over the sequence length for each sample.
        Finally, average over the batch.
        
        Args:
            logits (torch.Tensor): Logits from the model of shape [B, L, vocab_size].
            labels (torch.Tensor): Target labels of shape [B, L] with -100 for tokens that should be ignored.
        
        Returns:
            loss (torch.Tensor): Averaged loss over the batch (a scalar tensor).
        """
        loss_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='none',
            ignore_index=self.tokenizer.pad_token_id,
        )
        # Reshape to [B, L]
        loss_token = loss_token.view(labels.size(0), labels.size(1))
        valid_mask = labels != self.tokenizer.pad_token_id
        sample_loss = (loss_token * valid_mask.float()).sum(dim=1) / valid_mask.float().sum(dim=1).clamp(min=1)
        if loss_weight is not None:
            sample_loss *= loss_weight
        return sample_loss.mean()

    def training_step(self, batch, batch_idx):
        """
        Training step for the module.
        Expects the batch to be a dict with key 'input_ids'.
        """
        loss = self.step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    def step(self, batch):
        input_ids = batch["input_ids"]
        mask_ratio, masked_input_ids, labels = self.noise_x0(input_ids)
        logits = self(masked_input_ids)
        loss = self.compute_loss(logits, labels, 1/mask_ratio)
        return loss
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the module.
        Expects the batch to be a dict with key 'input_ids'.
        """
        loss = self.step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = get_optimizer(self.hparams.optimizer, self.model.parameters())
        if "lr_scheduler" in self.hparams and self.hparams.lr_scheduler is not None:
            lr_scheduler, extra_kwargs = get_scheduler(self.hparams.lr_scheduler, optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, **extra_kwargs},
            }
        return optimizer
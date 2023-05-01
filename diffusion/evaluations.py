from types import FunctionType

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
import pytorch_lightning as pl

from .diffusion import Diffusion
from .vae import SimpleVAE
from .unet import SimpleUNet
from .script_util import ModelType, VarType

class DiffusionEvaluator():
    def __init__(
        self,
        *,
        logger: FunctionType,
        dataloader: DataLoader=None,
        unet: SimpleUNet=None,
        diffusion: Diffusion=None,
        vae: SimpleVAE=None,
        classifier: nn.Module=None,
        batch_size: int=16,
    ):
        """
        logger[method pointer]: pl.LightningModule.log
        classifier: a classifier to test sample quality
        """
        self.logger = logger
        self.dataloader = dataloader
        self.unet = unet
        self.diffusion = diffusion
        self.vae = vae
        self.classifier = classifier
        self.batch_size = batch_size
        
        self.num_classes = self.diffusion.num_classes
        self.top_k_accs = [torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, top_k=k)
                           for k in range(self.num_classes)]
        
        # precomputing
        self.setup()

    def setup(self):
        """
        Calculates distribution of validation set
        """
        if self.vae is not None:
            # iterate over validation set and encode images
            zs, ys = list(), list()
            for x, y in iter(self.dataloader):
                zs.append(self.vae.get_latent(x))
                ys.append(y)
            zs, ys = torch.cat(zs), torch.cat(ys)
            
            # calculations used for FVAED score uncond
            self.encoded_val_set_mean = zs.mean(dim=0)
            self.encoded_val_set_cov = torch.cov(zs.T)
            
            # calculations used for FVAED score cond
            self.encoded_val_set_mean_class_wise = list()
            self.encoded_val_set_cov_class_wise = list()
            for label in range(self.num_classes):
                mask = ys == label
                indices = torch.nonzero(mask)
                zs_label_masked = zs[indices]
                self.encoded_val_set_mean_class_wise.append(zs_label_masked.mean(dim=0))
                self.encoded_val_set_cov_class_wise.append(torch.cov(zs_label_masked.T))

    def calculate_FVAED(self, mu1: torch.Tensor, cov1: torch.Tensor, mu2: torch.Tensor, cov2: torch.Tensor):
        sse = torch.sum(torch.square(mu1 - mu2))
        covmean = torch.sqrt(torch.matmul(cov1, cov2))
        return sse + torch.trace(cov1 + cov2 - 2 * covmean)

    def test_classifier_acc(self, n_tests: int):
        """
        uses the classifier to test sample quality of DDPM
        """
        if (self.unet.model_type is not ModelType.uncond) or (self.classifier is None) or (self.diffusion is None) or (self.unet is None):
            return
        # reset acc
        for acc in self.top_k_accs:
            acc.reset()
        # do n_tests and save accs
        while n_tests > 0:
            batch_size = min(n_tests, self.batch_size)
            img, y = self.diffusion.sample(self.unet, batch_size, y=True, show_pbar=False, to_img=False)
            img = (img.clamp(-1, 1) + 1) / 2
            y_preds = self.classifier.forward(img)
            for acc in self.top_k_accs:
                acc(y_preds, y)
            n_tests -= batch_size
        # log accs
        for k, acc in enumerate(self.top_k_accs, start=1):
            self.logger(f'top_{k}_acc', acc, on_epoch=True, prog_bar=False, logger=True)
    
    def test_FVAED(self, n_tests: int):
        """
        Like FID but uses vae insted of inception_v3
        """
        if (self.diffusion is None) or (self.unet is  None) or (self.vae is None):
            return
        
        cond = self.unet.model_type is ModelType.uncond
        sample = lambda batch_size: self.diffusion.sample(self.unet, batch_size, y=cond, show_pbar=False, to_img=False)
        
        # encode n_tests random samples
        zs, ys = list(), list()
        while n_tests > 0:
            batch_size = min(n_tests, self.batch_size)
            img, y = sample(batch_size)
            img = (img.clamp(-1, 1) + 1) / 2
            zs.append(self.vae.get_latent(img))
            ys.append(y)
            n_tests -= batch_size
        zs = torch.cat(zs)
        
        # compute and log FVAED
        FVAED = self.calculate_FVAED(
            zs.mean(dim=0),
            torch.cov(zs.T),
            self.encoded_val_set_mean,
            self.encoded_val_set_cov
        )
        self.logger(f'FVAED', FVAED, on_epoch=True, prog_bar=False, logger=True)
        
        # compute and log clasewise FVAED
        if self.unet.model_type is ModelType.uncond:
            for label in range(self.num_classes):
                ys = torch.cat(ys)
                mask = ys == label
                indices = torch.nonzero(mask)
                zs_label_masked = zs[indices]
                FVAED = self.calculate_FVAED(
                    zs_label_masked.mean(dim=0),
                    torch.cov(zs_label_masked.T),
                    self.encoded_val_set_mean_class_wise[label],
                    self.encoded_val_set_cov_class_wise[label]
                )
                self.logger(f'FVAED_{label}', FVAED, on_epoch=True, prog_bar=False, logger=True)
    
    def do_all_tests(self, n_tests: int):
        self.test_FVAED(n_tests)
        self.test_classifier_acc(n_tests)
from types import FunctionType

import numpy as np
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
import pytorch_lightning as pl

from .diffusion import Diffusion
from .vae import SimpleVAE
from .unet import SimpleUNet
from .script_util import ModelType, VarType

# TODO: Add confussion matrix
# TODO: only sample once

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
        n_sampling_tests: int=256
    ):
        """
        logger[method pointer]: pl.LightningModule.log
        classifier: a classifier to test sample quality
        """
        self.logger = logger
        self.dataloader = dataloader
        self.unet = unet.eval() if unet is not None else unet
        self.diffusion = diffusion
        self.vae = vae.eval() if vae is not None else unet
        self.classifier = classifier.eval() if classifier is not None else unet
        self.batch_size = batch_size
        self.n_sampling_tests = n_sampling_tests
        
        self.num_classes = self.diffusion.num_classes
        self.top_k_accs = [torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, top_k=k).to(vae.device)
                           for k in range(1, self.num_classes)]
        
        assert None not in [dataloader, vae, classifier], f'not initialized {[dataloader, vae, classifier]=}'
        
        self.is_prepared = False

    def setup(self, show_progress=False):
        """
        Calculates distribution of validation set
        """
        if show_progress:
            print("Preparing evaluator", flush=True)
        if self.vae is not None:
            # iterate over validation set and encode images
            zs, ys = list(), list()
            if show_progress:
                print('Encoding dataset', flush=True)
            iterable = tqdm.tqdm(iter(self.dataloader)) if show_progress else iter(self.dataloader)
            for x, y in iterable:
                zs.append(self.vae.get_latent(x.to(self.vae.device)))
                ys.append(y)
            zs, ys = torch.cat(zs), torch.cat(ys)
            
            # calculations used for FVAED score uncond
            self.encoded_val_set_mean = zs.mean(dim=0)
            self.encoded_val_set_cov = torch.cov(zs.T)
            
            # calculations used for FVAED score cond
            self.encoded_val_set_mean_class_wise = list()
            self.encoded_val_set_cov_class_wise = list()
            if show_progress:
                print('Label based computations', flush=True)
            iterable = tqdm.trange(self.num_classes) if show_progress else range(self.num_classes)
            for label in iterable:
                mask = ys == label
                indices = torch.nonzero(mask)
                zs_label_masked = zs[indices].squeeze()
                self.encoded_val_set_mean_class_wise.append(zs_label_masked.mean(dim=0))
                self.encoded_val_set_cov_class_wise.append(torch.cov(zs_label_masked.T))
        
        self.is_prepared = True
        self.has_sampled = False

    def calculate_FVAED(self, mu1: torch.Tensor, cov1: torch.Tensor, mu2: torch.Tensor, cov2: torch.Tensor):
        sse = torch.sum(torch.square(mu1 - mu2))
        covmean = torch.sqrt(torch.matmul(cov1, cov2))
        return sse + torch.trace(cov1 + cov2 - 2 * covmean)

    def test_classifier_acc(self, n_tests: int, show_progress=False):
        """
        uses the classifier to test sample quality of DDPM
        """
        if (self.unet.model_type is not ModelType.cond_embed) or (self.classifier is None) or (self.diffusion is None) or (self.unet is None):
            return
        # reset acc
        for acc in self.top_k_accs:
            acc.reset()
        # do n_tests and save accs
        if show_progress:
            print('Getting accuracy', flush=True)
            n_tests_start = n_tests
        while n_tests > 0:
            batch_size = min(n_tests, self.batch_size)
            cond = self.unet.model_type is ModelType.cond_embed
            img, y = self.diffusion.sample(self.unet, batch_size, y=cond, show_pbar=False, to_img=False)
            img = (img.clamp(-1, 1) + 1) / 2
            y_preds = self.classifier.forward(img)
            for acc in self.top_k_accs:
                acc(y_preds, y)
            n_tests -= batch_size
            if show_progress:
                print(f'{n_tests_start-n_tests}/{n_tests_start} left: {100*(n_tests_start-n_tests)/n_tests_start:.3}%', flush=True)
        # log accs
        for k, acc in enumerate(self.top_k_accs, start=1):
            self.logger(f'top_{k}_acc', acc.compute(), on_epoch=True, prog_bar=False, logger=True)
    
    def test_FVAED(self, n_tests: int, show_progress=False):
        """
        Like FID but uses vae insted of inception_v3
        """
        if (self.diffusion is None) or (self.unet is  None) or (self.vae is None):
            return
        
        if not self.is_prepared:
            self.setup(show_progress=show_progress)
        
        cond = self.unet.model_type is ModelType.cond_embed
        sample = lambda batch_size: self.diffusion.sample(self.unet, batch_size, y=cond, show_pbar=False, to_img=False)
        
        if show_progress:
            print('Computing FVAED', flush=True)
            n_tests_start = n_tests
        # encode n_tests random samples
        zs, ys = list(), list()
        while n_tests > 0:
            batch_size = min(n_tests, self.batch_size)
            img, y = sample(batch_size)
            img = (img.clamp(-1, 1) + 1) / 2
            zs.append(self.vae.get_latent(img))
            ys.append(y)
            n_tests -= batch_size
            if show_progress:
                print(f'{n_tests_start-n_tests}/{n_tests_start} left: {100*(n_tests_start-n_tests)/n_tests_start:.3}%', flush=True)
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
        if self.unet.model_type is ModelType.cond_embed:
            ys = torch.cat(ys)
            if show_progress:
                print('Computing class based FVAD', flush=True)
            iterable = tqdm.trange(self.num_classes) if show_progress else range(self.num_classes)
            for label in iterable:
                mask = ys == label
                indices = torch.nonzero(mask)
                zs_label_masked = zs[indices].squeeze()
                FVAED = self.calculate_FVAED(
                    zs_label_masked.mean(dim=0),
                    torch.cov(zs_label_masked.T),
                    self.encoded_val_set_mean_class_wise[label],
                    self.encoded_val_set_cov_class_wise[label]
                )
                self.logger(f'FVAED_{label}', FVAED, on_epoch=True, prog_bar=False, logger=True)
    
    def get_samples(self, n_tests: int=None):
        n_tests = n_tests or self.n_sampling_tests
        if self.has_sampled is False:
            pass # do samples
        self.has_sampled = True
    
    def do_all_tests(self, n_tests: int=None, show_progress=False):
        n_tests = n_tests or self.n_sampling_tests
        self.has_sampled = False
        with torch.no_grad():
            if show_progress:
                print("Starting classifier acc test", flush=True)
            self.test_classifier_acc(n_tests=n_tests, show_progress=show_progress)
            if not self.is_prepared:
                self.setup(show_progress=show_progress)
            if show_progress:
                print("Starting FVAED test", flush=True)
            self.test_FVAED(n_tests=n_tests, show_progress=show_progress)
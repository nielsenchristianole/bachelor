from types import FunctionType, LambdaType

import numpy as np
import tqdm
import math as m
import os

from sklearn.metrics import confusion_matrix

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
        n_sampling_tests: int=512,
        num_classes: int=10,
        save_path: LambdaType=lambda: './',
        assert_initialization: bool=True,
    ):
        """
        logger[method pointer]: e.g. pl.LightningModule.log
        dataloader: used for testing against when calculating FID
        classifier: a classifier to test sample quality
        vae: vae for replacing inception classifier in FID
        batch_size: working batch size, purely performance based
        n_sampling_tests: how many samples to compute when calculating metrics
        num_classes: number of classes
        save_path: where to save extra information, e.g. conf matrix
        assert_initialization: test of initialized with all needed items
        """
        self.logger = logger
        self.dataloader = dataloader
        self.unet = unet.eval() if unet is not None else unet
        self.diffusion = diffusion
        self.vae = vae.eval() if vae is not None else unet
        self.classifier = classifier.eval() if classifier is not None else unet
        self.batch_size = batch_size
        self.n_sampling_tests = n_sampling_tests
        self.num_classes = num_classes
        self.save_path = save_path
        self.assert_initialization = assert_initialization
        
        self.top_k_accs = [torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=k).to(vae.device)
                           for k in range(1, num_classes)]
        
        assert (not assert_initialization) or (None not in [dataloader, vae, classifier]), f'not initialized {[dataloader, vae, classifier]=}'
        self.is_prepared = False
        self.has_sampled = False

    def prepare(self, show_progress=False):
        """
        Calculates and store computations only needed once for multiple tests
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

    def compute_and_save_conf_matrix(self, y_true: list[torch.Tensor], y_pred: list[torch.Tensor]):
        """
        takes a list of predictions (logits or probs) and true labels and saves conf matrix to self.save_path
        """
        # process preds
        y_true = torch.cat(y_true).detach().cpu().numpy()
        y_pred = torch.argmax(torch.cat(y_pred), dim=1).detach().cpu().numpy()
        # get name of save file
        save_location = lambda normalize: self.save_path() + f'conf_matrix_{normalize}.npy'
        for normalize in (None, 'true'):
            conf_matrix = confusion_matrix(y_true, y_pred, normalize=normalize)
            with open(save_location(normalize or 'count'), 'wb') as f:
                np.save(f, conf_matrix)

    def test_classifier_acc(self, n_tests: int, show_progress=False):
        """
        classifies the generated images and store accuracies and conf matrix
        """
        if (self.unet.model_type is not ModelType.cond_embed) or (self.classifier is None) or (self.diffusion is None) or (self.unet is None):
            return
        # reset acc
        for acc in self.top_k_accs:
            acc.reset()
        # do n_tests and save accs
        if show_progress:
            print('Getting accuracy', flush=True)
        iterator = list(zip(*self.get_samples(n_tests=n_tests, show_progress=show_progress))) # list for tqdm
        all_y_true, all_y_preds = list(), list()
        for img, y in tqdm.tqdm(iterator) if show_progress else iterator:
            img = (img.clamp(-1, 1) + 1) / 2
            y_preds = self.classifier.forward(img)
            for acc in self.top_k_accs:
                acc(y_preds, y)
            all_y_true.append(y)
            all_y_preds.append(y_preds)
        self.compute_and_save_conf_matrix(all_y_true, all_y_preds)
        # log accs
        for k, acc in enumerate(self.top_k_accs, start=1):
            self.logger(f'top_{k}_acc', acc.compute(), on_epoch=True, prog_bar=False, logger=True)

    def calculate_FVAED(self, mu1: torch.Tensor, cov1: torch.Tensor, mu2: torch.Tensor, cov2: torch.Tensor):
        sse = torch.sum(torch.square(mu1 - mu2))
        covmean = torch.sqrt(torch.matmul(cov1, cov2))
        return sse + torch.trace(cov1 + cov2 - 2 * torch.sqrt(covmean))
    
    def test_FVAED(self, n_tests: int=None, show_progress=False):
        """
        Like FID but uses vae insted of inception_v3
        Does calculation for whole data distribution and individual labels
        """
        if (self.diffusion is None) or (self.unet is  None) or (self.vae is None):
            return
        
        if not self.is_prepared:
            self.prepare(show_progress=show_progress)
        
        if show_progress:
            print('Computing FVAED', flush=True)
        # encode n_tests random samples
        xs, ys = self.get_samples(n_tests=n_tests, show_progress=show_progress)
        zs = list()
        for img in tqdm.tqdm(xs) if show_progress else xs:
            img = (img.clamp(-1, 1) + 1) / 2
            zs.append(self.vae.get_latent(img))
        zs = torch.cat(zs)
        
        # compute and log FVAED
        FVAED = self.calculate_FVAED(
            zs.mean(dim=0),
            torch.cov(zs.T),
            self.encoded_val_set_mean,
            self.encoded_val_set_cov
        )
        self.logger(f'FVAED', FVAED, on_epoch=True, prog_bar=False, logger=True)
        
        # compute and log classwise FVAED
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
    
    def get_samples(self, n_tests: int=None, show_progress=False) -> tuple[list[torch.Tensor], list[torch.Tensor|None]]:
        """
        samples once and stores for any computation
        """
        n_tests = n_tests or self.n_sampling_tests
        if self.has_sampled is False:
            self.has_sampled = True
            self.xs = list[torch.Tensor]()
            self.ys = list[torch.Tensor|None]()
            batch_max_size = m.ceil(n_tests / self.batch_size)
            y_labels = np.split(np.arange(n_tests) % self.num_classes, batch_max_size)
            if show_progress:
                print('Starting sampling')
                print('batch_sizes:', ', '.join([str(len(a)) for a in y_labels]))
            for y_label in tqdm.tqdm(y_labels) if show_progress else y_labels:
                batch_size = len(y_label)
                y_label = y_label if self.unet.model_type is ModelType.cond_embed else None
                img, y = self.diffusion.sample(self.unet, batch_size, y=y_label, show_pbar=False, to_img=False)
                self.xs.append(img)
                self.ys.append(y)
        return self.xs, self.ys
    
    def set_samples(self, xs: list[torch.Tensor], ys: list[torch.Tensor]|None=None):
        """
        This function is to set the samples used for testing, e.g. when calculating FVEAD on VAE reconstructions
        """
        self.has_sampled = True
        self.xs = xs
        self.ys = ys
    
    def do_all_tests(self, n_tests: int=None, show_progress=False):
        """
        Is only meant to be used once for each model e.g. when when training on epoch end
        Reset samples and does all above tests
        """
        n_tests = n_tests or self.n_sampling_tests
        self.has_sampled = False
        with torch.no_grad():
            if show_progress:
                print("Starting classifier acc test", flush=True)
            self.test_classifier_acc(n_tests=n_tests, show_progress=show_progress)
            if not self.is_prepared:
                self.prepare(show_progress=show_progress)
            if show_progress:
                print("Starting FVAED test", flush=True)
            self.test_FVAED(n_tests=n_tests, show_progress=show_progress)
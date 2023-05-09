from types import FunctionType, LambdaType

import numpy as np
import tqdm
import math as m
import os

from sklearn.metrics import confusion_matrix
from scipy import linalg

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
import pytorch_lightning as pl

from .diffusion import Diffusion
from .vae import SimpleVAE
from .unet import SimpleUNet
from .script_util import ModelType, VarType


def calculate_FVAED(sampled_mu, sampled_sigma, true_mu, true_sigma, *, eps=1e-6):
    """
    inputs - mu1: torch.Tensor, cov1: torch.Tensor, mu2: torch.Tensor, cov2: torch.Tensor
    Code inspired from https://github.com/mseitzer/pytorch-fid
    """
    to_numpy = lambda tensor: tensor.detach().cpu().numpy()
    mu1, sigma1, mu2, sigma2 = list(map(to_numpy, (sampled_mu, sampled_sigma, true_mu, true_sigma)))
    
    # Product might be almost singular
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


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
            y_preds = self.classifier.forward(img)
            for acc in self.top_k_accs:
                acc(y_preds, y)
            all_y_true.append(y)
            all_y_preds.append(y_preds)
        self.compute_and_save_conf_matrix(all_y_true, all_y_preds)
        # log accs
        for k, acc in enumerate(self.top_k_accs, start=1):
            self.logger(f'top_{k}_acc', acc.compute(), on_epoch=True, prog_bar=False, logger=True)
    
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
            zs.append(self.vae.get_latent(img))
        zs = torch.cat(zs)
        
        # compute and log FVAED
        FVAED = calculate_FVAED(
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
                FVAED = calculate_FVAED(
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
                img = img.clamp(0., 1.)
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
        self.xs = None
        self.ys = None
        torch.cuda.empty_cache()


class CounterfactEvaluator():
    def __init__(
        self,
        *,
        data_module: pl.LightningDataModule,
        combined_model: pl.LightningModule,
        vae: SimpleVAE,
        classifier: nn.Module,
        experiments: dict[int, list[int]],
        n_classes: int=10,
        show_progress: bool=True,
        batch_size: int=16,
        save_path: LambdaType=lambda: './',
        save_results: bool=True
    ):
        """
        This class generates counterfactual images for a dataloader taking in experiments.
        If wanting to do two experiments
        1) transform 6 into 0 and 5
        2) transform 4 into 9
        set experiments={6: [0, 5], 4: [9]}
        """
        self.data_module = data_module
        self.combined_model = combined_model
        self.vae = vae.eval()
        self.classifier = classifier.eval()
        self.experiments = experiments
        self.n_classes = n_classes
        self.show_progress = show_progress
        self.batch_size = batch_size
        self.save_path = save_path
        self.save_results = save_results
        
        self.unet: SimpleUNet
        self.diffusion: Diffusion
        self.unet, self.diffusion = combined_model.extract_models()
        self.data_module.prepare_data()
        self.has_sampled = False
        self.sampled_counterfactuals = None
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes).to(classifier.device)
        
    def prepare_dataloader_FVAED(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates and store computations only needed once for multiple tests
        """
        # iterate over validation set and encode images
        zs, ys = list(), list()
        if self.show_progress:
            print('Encoding dataset', flush=True)
        iterable = tqdm.tqdm(iter(dataloader), "Preparing FVAED calc") if self.show_progress else iter(dataloader)
        for x, y in iterable:
            zs.append(self.vae.get_latent(x.to(self.vae.device)))
            ys.append(y)
        zs, ys = torch.cat(zs), torch.cat(ys)
        
        # calculations used for FVAED score uncond
        mean = zs.mean(dim=0)
        cov = torch.cov(zs.T)
        return mean, cov
    
    def sample_counterfacts(
        self,
        dataloader: DataLoader=None,
        *,
        y_target: int=None,
        tau: int=None,
        lambda_p: float=None,
        lambda_c: float=None,
        vgg_block: int=None,
        n_reconstruct_samples: int=None,
    ) -> list[torch.Tensor]:
        """
        tau: the depth the image should be noised to
        lambda_p: the perception loss weight (how much it should look like the original)
        lamdba_c: the classification loss weight (how much it should look like the counterfactual label)
        vgg_block: how far into the vgg net the perception loss is calculated
        n_reconstruct_samples: how many times it should sample the gradiant unconditionally when doing guided sampling
        """
        if not self.has_sampled:
            self.sampled_counterfactuals = list()
            iterator = tqdm.tqdm(iter(dataloader), "Sampling counterfactuals") if self.show_progress else iter(dataloader)
            for x_0, _ in iterator:
                counterfact, _ = self.diffusion.guided_counterfactual(
                    model=self.unet,
                    classifier=self.classifier,
                    x_0=x_0.to(self.classifier.device),
                    y=y_target,
                    tau=tau,
                    lambda_p=lambda_p,
                    lambda_c=lambda_c,
                    vgg_block=vgg_block,
                    n_reconstruct_samples=n_reconstruct_samples,
                    show_pbar=False
                )
                self.sampled_counterfactuals.append(counterfact.detach())
            self.has_sampled = True
        return self.sampled_counterfactuals
        
    def calculate_pair_wise_FVAED(self, stage='fit'):
        
        means = list()
        covs = list()
        
        self.data_module.prepare_data()
        
        _range = tqdm.trange if self.show_progress else range
        for label in _range(self.n_classes):
            self.data_module.label_subset = [label]
            self.data_module.setup(stage)
            if stage == 'fit':
                dataloader = self.data_module.val_dataloader()
            elif stage == 'test':
                dataloader = self.data_module.test_dataloader()
            else:
                raise NotImplementedError(f'{stage=} not implemented')
            mean, cov = self.prepare_dataloader_FVAED(dataloader)
            means.append(mean)
            covs.append(cov)
        
        return_matrix = np.full((self.n_classes, self.n_classes), 0.)
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if j == i:
                    continue
                return_matrix[i, j] = calculate_FVAED(means[i], covs[i], means[j], covs[j])
        
        return return_matrix

    def do_all_tests(
        self,
        stage='fit',
        *,
        tau: int,
        lambda_p: float,
        lambda_c: float,
        vgg_block: int,
        n_reconstruct_samples: int
        ):
        """
        stage: which dataset to use fit=validation, test=test
        returns a dict with results
        """
        return_dict = dict()
        
        self.data_module.prepare_data()
        
        counterfact_FVAED = np.full((self.n_classes, self.n_classes), -1.)
        counterfact_acc = np.full((self.n_classes, self.n_classes), -1.)
        for original_label, target_labels in self.experiments.items():
            self.data_module.label_subset = [original_label]
            self.data_module.setup(stage)
            if stage == 'fit':
                dataloader = self.data_module.val_dataloader()
            elif stage == 'test':
                dataloader = self.data_module.test_dataloader()
            else:
                raise NotImplementedError(f'{stage=} not implemented')
        
            true_mean, true_cov = self.prepare_dataloader_FVAED(dataloader)
        
            for target_label in target_labels:
                self.has_sampled = False
                # calculate counterfact FVAED
                sampled_counterfactuals = self.sample_counterfacts(
                    dataloader=dataloader,
                    y_target=target_label,
                    tau=tau,
                    lambda_p=lambda_p,
                    lambda_c=lambda_c,
                    vgg_block=vgg_block,
                    n_reconstruct_samples=n_reconstruct_samples
                )
                zs = list()
                for sample in tqdm.tqdm(sampled_counterfactuals, "Calculating FVAED") if self.show_progress else sampled_counterfactuals:
                    zs.append(self.vae.get_latent(sample))
                zs = torch.cat(zs)
                counterfact_mean = zs.mean(dim=0)
                counterfact_cov = torch.cov(zs.T)
                print(counterfact_mean, counterfact_cov, true_mean, true_cov, flush=True)
                fvead = calculate_FVAED(counterfact_mean, counterfact_cov, true_mean, true_cov)
                counterfact_FVAED[original_label, target_label] = fvead
                
                # calculate prediction
                self.acc.reset()
                for sample in tqdm.tqdm(sampled_counterfactuals, "Calculating acc") if self.show_progress else sampled_counterfactuals:
                    y_preds = self.classifier.forward(sample)
                    y_target = torch.full((len(y_preds),), target_label, device=y_preds.device)
                    self.acc(y_preds, y_target)
                counterfact_acc[original_label, target_label] = float(self.acc.compute())
            
            return_dict['counterfact_FVAED'] = counterfact_FVAED
            return_dict['counterfact_acc'] = counterfact_acc
            
            if self.save_results:
                np.save(os.path.join(self.save_path(), 'counterfact_FVAED.npy'), counterfact_FVAED)
                np.save(os.path.join(self.save_path(), 'counterfact_acc.npy'), counterfact_acc)
        
        return return_dict
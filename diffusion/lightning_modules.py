import os

from numpy import random

import torch
from torch import nn
import pytorch_lightning as pl

from types import FunctionType

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, Subset

from .diffusion import Diffusion
from .unet import SimpleUNet
from .script_util import ModelType, VarType
from .evaluations import DiffusionEvaluator


class DiffusionWithModel(pl.LightningModule):
    def __init__(
        self,
        params: dict,
        *,
        evaluator: DiffusionEvaluator|None=None
    ):
        """
        This is a module used purely for training and testing the models
        classifier: only used for testing during training
        encoder: only used for testing during training
        var_dataloader: only used for testing during training
        """
        super().__init__()
        
        self.save_hyperparameters(ignore=('encoder', 'classifier', 'var_dataloader'))
        
        self.model_type = params.get('model_type')
        self.var_type = params.get('var_type')
        self.lambda_vlb_weight = params.get('lambda_vlb_weight')
        self.model = SimpleUNet(
            model_type=self.model_type,
            var_type=self.var_type,
            **params.get('unet_kwargs')
        )
        self.diffusion = Diffusion(**params.get('diffusion_kwargs'))
        self.params = params
        self.MSEloss = nn.MSELoss()
    
        self.evaluator = evaluator
    
    def forward(self, x):
        return self.model.forward(x)
    
    def process_batch(self, batch, vlb_logger: FunctionType=None):
        x_0, y = batch
        t = self.diffusion.sample_timesteps(x_0.shape[0])
        x_t, noise = self.diffusion.q_sample(x_0, t)
        # trains unconditionally 10% of the time
        if self.model_type is ModelType.cond_embed and random.rand() < 0.1:
            y = None
        model_out = self.model.forward(x_t, t, y)
        if self.var_type is VarType.scheduled:
            loss = self.MSEloss(model_out, noise)
        elif self.var_type is VarType.learned:
            err, v = self.diffusion._split_model_out(model_out, x_t.shape)
            loss_simple = self.MSEloss(err, noise)
            # variational lower bound
            loss_vlb = self.lambda_vlb_weight * self.diffusion.calculate_loss_vlb(err.detach(), v, x_0, x_t, t) # should not use D_kl to train mean prediction
            (vlb_logger is None) or vlb_logger(loss_vlb) # log if logger is provided
            loss = loss_simple + loss_vlb
        else:
            raise NotImplementedError(f'Unknown {self.var_type=}')
        return loss

    def training_step(self, batch, batch_idx):
        vlb_logger = lambda loss_vlb: self.log('train_vlb_loss', loss_vlb, on_step=True, prog_bar=False, logger=True)
        loss = self.process_batch(batch, vlb_logger=vlb_logger)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        vlb_logger = lambda loss_vlb: self.log('val_vlb_loss', loss_vlb, on_epoch=True, prog_bar=False, logger=True)
        loss = self.process_batch(batch, vlb_logger=vlb_logger)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        vlb_logger = lambda loss_vlb: self.log('test_vlb_loss', loss_vlb, on_epoch=True, prog_bar=False, logger=True)
        loss = self.process_batch(batch, vlb_logger=vlb_logger)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_train_epoch_end(self) -> None:
        """
        Does a full evaluation on the end of every training epoch using the evaluator
        """
        if self.evaluator is None:
            return
        if self.evaluator.is_prepared is False:
            self.evaluator.prepare()
        self.evaluator.unet, self.evaluator.diffusion = self.extract_models()
        self.evaluator.do_all_tests()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params.get('lr'))
    
    def extract_models(self) -> tuple[SimpleUNet, Diffusion]:
        return self.model.to(self.device).eval(), self.diffusion
    

class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=None,
        num_workers=None,
        normalize=True,
        train_val_split=[55000, 5000],
        data_split_seed=42,
        label_subset: list[int]|None=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = os.cpu_count() if num_workers is None else num_workers
        self.normalize = normalize
        self.train_val_split = train_val_split
        self.data_split_seed = data_split_seed
        self.label_subset = label_subset
        
        self.normelization = transforms.Normalize((0.,), (1.,)) if normalize else nn.Identity() # transforms.Normalize((0.1307,), (0.3081,)),
        
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                self.normelization
            ]
        )
    
    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def remove_subset_complement(self, dataset: MNIST|Subset):
        """
        returns a subset of the dataset containing only labels in self.label_subset
        """
        if self.label_subset is not None:
            if isinstance(dataset, Subset):
                mask = torch.full((len(dataset.indices),), False, dtype=bool)
                for label in self.label_subset:
                    mask |= (dataset.dataset.targets[dataset.indices] == label)
            elif isinstance(dataset, MNIST):
                mask = torch.full_like(dataset.targets, False, dtype=bool)
                for label in self.label_subset:
                    mask |= (dataset.targets == label)
            else:
                raise NotImplementedError(f'{type(dataset)=} not supported')
            indices = torch.arange(len(mask))[mask]
            return Subset(dataset, indices)
        return dataset

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            mnist_train, mnist_val = random_split(mnist_full, self.train_val_split, generator=torch.Generator().manual_seed(self.data_split_seed))
            self.mnist_train = self.remove_subset_complement(mnist_train)
            self.mnist_val = self.remove_subset_complement(mnist_val)
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            self.mnist_test = self.remove_subset_complement(mnist_test)
            
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
import os

import torch
from torch import nn

from numpy import random

import pytorch_lightning as pl

from .diffusion import Diffusion
from .unet import SimpleUNet
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

from .script_util import ModelType, VarType


class DiffusionWithModel(pl.LightningModule):
    def __init__(
        self,
        params: dict,
        loss_fn: nn.Module=None,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model_type = params.get('model_type')
        self.var_type = params.get('var_type')
        self.model = SimpleUNet(
            model_type=self.model_type,
            var_type=self.var_type,
            **params.get('unet_kwargs')
        )
        self.diffusion = Diffusion(**params.get('diffusion_kwargs'))
        self.params = params
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
    
    def process_batch(self, batch):
        x, y = batch
        t = self.diffusion.sample_timesteps(x.shape[0])
        x_t, noise = self.diffusion.q_sample(x, t)
        # trains unconditionally 10% of the time
        if self.model_type is ModelType.cond_embed and random.rand() < 0.1:
            y = None
        model_out = self.model.forward(x_t, t, y)
        if self.var_type is VarType.learned:
            raise NotImplementedError
        else:
            loss = self.loss_fn(model_out, noise)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params.get('lr'))
    
    def extract_models(self):
        return self.model.to(self.device), self.diffusion
    

class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers=None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = os.cpu_count() if num_workers is None else num_workers
        
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.,), (1.,)) # transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    
    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
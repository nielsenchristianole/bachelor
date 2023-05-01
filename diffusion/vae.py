import torch

import torch
from torch import nn
from torchvision import transforms

import numpy as np

import math as m

import pytorch_lightning as pl


const = m.log(2 * m.pi) / 2
get_log_prob = lambda x, mu, log_std: -log_std - const - torch.square((x - mu) / torch.exp(log_std)) / 2
reduce = lambda x: torch.sum(x.view(x.size(0), -1), dim=1)


class SimpleVAE(pl.LightningModule):
    def __init__(
        self,
        input_dims=(1,28,28),
        latens_dim=2,
        dropout=0.,
        global_pooling: nn.Module=nn.MaxPool2d,
        lr=1e-4,
        training_normelization=True,
        beta=1.,
        use_simple=False
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=[global_pooling])
        
        self.input_dims = input_dims
        self.latens_dim = latens_dim
        self.dropout = dropout
        self.lr = lr
        self.training_normelization = training_normelization
        self.beta = beta
        self.use_simple = use_simple
        
        prod_spacial = np.prod(input_dims)
        
        in_channels, *spacial = input_dims
        global_pool_kernal_dim = [d // 2 // 2 for d in spacial]
        
        self.normelize = transforms.Normalize((0.,), (1.,)) if training_normelization else nn.Identity()
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, (3,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3,3), padding='same'),
            nn.MaxPool2d((2,2)),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3,3), padding='same'),
            nn.MaxPool2d((2,2)),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3), padding='same'),
            global_pooling(global_pool_kernal_dim) # global max pooling
        )
        
        self.encoder_dense = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, 2 * latens_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latens_dim, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, prod_spacial),
        )
        
        self.simple_encoder = nn.Sequential(
            nn.Linear(in_features=prod_spacial, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2*latens_dim)
        )
        self.simple_decoder = nn.Sequential(
            nn.Linear(in_features=latens_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=prod_spacial)
        )
        
        self.loss_fn = nn.BCELoss()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_simple:
            x = torch.flatten(x, 1)
            return(self.simple_encoder(x))
        x = self.encoder_cnn(x)
        x = torch.flatten(x, 1)
        return self.encoder_dense(x)

    def split_mean_log_std(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(torch.split(x, 2, 1))
    
    def get_latent(self, x: torch.Tensor):
        z = self.encode(x)
        mu, log_std = self.split_mean_log_std(z)
        return mu
    
    def reparameterize(self, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_simple:
            x = self.simple_decoder(z)
        else:
            x = self.decoder(z)
        x = x.view(-1, *self.input_dims)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encode(x)
        mu, log_std = self.split_mean_log_std(x)
        x = self.reparameterize(mu, log_std)
        x = self.decode(x)
        return x
    
    def process_batch(self, batch):
        x, y = batch
        h = self.normelize(x)
        
        h = self.encode(h)
        mu, log_std = self.split_mean_log_std(h)
        z = self.reparameterize(mu, log_std)
        
        x_pred = self.decode(z)
        
        log_px = reduce(torch.distributions.bernoulli.Bernoulli(logits=x_pred, validate_args=False).log_prob(x))
        log_pz = reduce(get_log_prob(z, torch.zeros_like(z), torch.zeros_like(z))) # prior
        log_qz = reduce(get_log_prob(z, mu, log_std)) # posterior
        
        kl = log_qz - log_pz
        beta_elbo = log_px - self.beta * kl
        
        loss = -beta_elbo.mean()
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
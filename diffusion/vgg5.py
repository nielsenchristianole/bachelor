import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics

from collections.abc import Sequence


class N_conv_pool_block(nn.Module):
    """
    Simple vgg block consisting of n serian blocks of Conv2d, Batchnorm and ReLU activation.add()
    channels: a sequence of ints, where the first in in_dims, last is out_dims and those inbetween are middle_dims
    pool_padding, if dim/2 is uneven, add polling to make it even, then future poolings don't leave spacial information out
    """
    def __init__(self, channels: Sequence[int], pool_padding=(0, 0)):
        super().__init__()
        
        assert len(channels) >= 2, "Has to specify at least two values for channels (in and out)"

        in_channel = channels[0]
        self.convblocks = nn.ModuleList([])
        for out_channel in channels[1:]:
            convblock = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, (3,3), padding='same'),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )
            self.convblocks.append(convblock)
            in_channel = out_channel
        self.pooling = nn.MaxPool2d((2,2), padding=pool_padding)
        
    def forward(self, x: torch.Tensor, use_pooling: bool=True):
        for module in self.convblocks:
            x = module(x)
        if use_pooling:
            x = self.pooling(x)
        return x


class VGG5(pl.LightningModule):
    """
    Adopted from https://github.com/kkweon/mnist-competition/blob/master/vgg5.py with few changes
    calculating the img_shape between blocks to determine need of pooling and dim og the dense layers
    """
    def __init__(self, img_size=(28, 28), in_channels=1, num_classes=10, lr=1e-3, dropout=0.5):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        img_size = np.array(img_size) # change to np.array for easier array manipulation
        
        img_size //= 2 # AvgPool
        padding = img_size % 2
        img_size += padding
        self.block_1 = N_conv_pool_block((in_channels, 32, 32), pool_padding=padding.tolist())
        
        img_size //= 2 # AvgPool
        padding = img_size % 2
        img_size += padding
        self.block_2 = N_conv_pool_block((32, 64, 64), pool_padding=padding.tolist())
        
        img_size //= 2 # AvgPool
        padding = img_size % 2
        img_size += padding
        self.block_3 = N_conv_pool_block((64, 128, 128, 128), pool_padding=padding.tolist())
        
        img_size //= 2 # AvgPool
        padding = img_size % 2
        img_size += padding
        self.block_4 = N_conv_pool_block((128, 256, 256, 256), pool_padding=padding.tolist())
        
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(256 * img_size.prod(), 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    
    def class_distribution(self, x):
        """
        Apply softmax to out logits of the forward pass
        """
        x = self.forward(x)
        y_dist = F.softmax(x, dim=1)
        return y_dist
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply argmax to out logits of the forward pass
        """
        x = self.forward(x)
        y_pred = torch.argmax(x, dim=1)
        return y_pred
    
    def get_featuremap(self, x: torch.Tensor, block: int):
        """
        Get the featuremap output from one of the blocks without pooling, 
        use normal forward untill block is found
        """
        assert block in {1, 2, 3, 4}, f"{block=} not supported"
        x = self.block_1(x, use_pooling=block!=1)
        if block == 1:
            return x
        x = self.block_2(x, use_pooling=block!=2)
        if block == 2:
            return x
        x = self.block_3(x, use_pooling=block!=3)
        if block == 3:
            return x
        return self.block_4(x, use_pooling=False)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        
        self.train_acc(y_pred, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, logger=True)
        
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        
        self.val_acc(y_pred, y)
        self.log('valid_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        
        self.test_acc(y_pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        
        loss = self.loss_fn(y_pred, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
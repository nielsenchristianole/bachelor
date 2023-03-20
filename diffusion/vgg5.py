import torch
from torch import nn

import pytorch_lightning as pl


class Two_conv_pool(nn.Module):
    def __init__(self, in_channels, mid_channels_1, mid_channels_2):
        super().__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels_1, (3,3), padding='same'),
            nn.BatchNorm2d(mid_channels_1),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(mid_channels_1, mid_channels_2, (3,3), padding='same'),
            nn.BatchNorm2d(mid_channels_2),
            nn.ReLU()
        )
        self.pooling = nn.MaxPool2d((2,2))
    
    def forward(self, x: torch.Tensor, use_pooling: bool=True):
        x = self.conv_1(x)
        x = self.conv_2(x)
        if use_pooling:
            x = self.pooling(x)
        return x
        

class Three_conv_pool(nn.Module):
    def __init__(self, in_channels, mid_channels_1, mid_channels_2, mid_channels_3):
        super().__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels_1, (3,3), padding='same'),
            nn.BatchNorm2d(mid_channels_1),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(mid_channels_1, mid_channels_2, (3,3), padding='same'),
            nn.BatchNorm2d(mid_channels_2),
            nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(mid_channels_2, mid_channels_3, (3,3), padding='same'),
            nn.BatchNorm2d(mid_channels_3),
            nn.ReLU()
        )
        self.pooling = nn.MaxPool2d((2,2))
    
    def forward(self, x: torch.Tensor, *, use_pooling: bool=True):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        if use_pooling:
            x = self.pooling(x)
        return x


class VGG5(pl.LightningModule):
    """
    Adopted from https://github.com/kkweon/mnist-competition/blob/master/vgg5.py
    """
    def __init__(self, in_channels=1, n_classes=10, loss_fn=None, lr=1e-3):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        
        self.block_1 = Two_conv_pool(in_channels, 32, 32)
        self.block_2 = Two_conv_pool(32, 64, 64)
        self.block_3 = Three_conv_pool(64, 128, 128, 128)
        self.block_4 = Three_conv_pool(128, 256, 256, 256)
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes)
        )
        
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = loss_fn
    
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    
    def predict(self, x):
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
    
    def process_batch(self, batch):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
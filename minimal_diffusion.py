import os

import torch
from torch import nn, utils

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from diffusion.script_util import diffusion_uncond_defaults, diffusion_cond_embed_defaults
from diffusion.lightning_modules import DiffusionWithModel, MNISTDataModule


def main(params: dict, seed=None, device_name=None, work_dir='./'):
    data_dir = work_dir
    work_dir = os.path.join(work_dir, params['model_type'].name)
    
    if device_name is None:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
        
    pl.seed_everything(seed, workers=True)
    
    data_module = MNISTDataModule(data_dir=data_dir, batch_size=params.get('batch_size'))
    channels, width, height = data_module.dims
    assert width == height, f"{width=}, {height=} image not square"
    params['diffusion_kwargs']['num_classes'] = params['unet_kwargs']['num_classes'] = data_module.num_classes
    params['diffusion_kwargs']['img_size'] = params['unet_kwargs']['img_size'] = width
    params['diffusion_kwargs']['color_channels'] = params['unet_kwargs']['out_channels'] = params['unet_kwargs']['in_channels'] = channels
    
    combined_model = DiffusionWithModel(params).to(device)

    loss_precesion = 6
    loss_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=3, filename='loss-{epoch}-{val_loss:.%sf}' % loss_precesion)
    epoch_callback = ModelCheckpoint(every_n_epochs=2, filename='epoch-{epoch}-{val_loss:.%sf}' % loss_precesion)
    last_callback = ModelCheckpoint(save_last=True, filename='last-{epoch}-{val_loss:.%sf}' % loss_precesion)
    
    trainer = pl.Trainer(
        default_root_dir=work_dir,
        callbacks=[loss_callback, epoch_callback, last_callback],
        log_every_n_steps=10,
        max_epochs=params.get('epochs'),
        logger=CSVLogger(work_dir, flush_logs_every_n_steps=100),
        strategy='dp', # data parallel
        accelerator=device_name,
        devices=torch.cuda.device_count() if device_name == 'cuda' else os.cpu_count()
    )
    
    trainer.fit(
        combined_model,
        data_module
    )

if __name__ == '__main__':
    for params in [diffusion_cond_embed_defaults(), diffusion_uncond_defaults()]:
        params['epochs'] = 1
        main(params, seed=42, work_dir='C:/Users/niels/local_data/bachelor')
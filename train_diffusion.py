import os
import argparse

import torch
from torch import nn, utils

from numpy import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from diffusion.script_util import get_hardware_kwargs, diffusion_uncond_simple
from diffusion.lightning_modules import DiffusionWithModel, MNISTDataModule


def main(
    params: dict,
    seed: int,
    *,
    hardware_kwargs: dict,
    experiment_name: str='unknown'
):
    work_dir = hardware_kwargs.pop('work_dir')
    data_dir = os.path.join(work_dir, 'datasets')
    models_dir = os.path.join(work_dir, 'models', f'{experiment_name}-{seed}')
    
    device = torch.device(hardware_kwargs.pop('device_name'))
    
    pl.seed_everything(seed, workers=True)
    
    data_module = MNISTDataModule(
        data_dir=data_dir,
        batch_size=hardware_kwargs.pop('batch_size'),
        num_workers=hardware_kwargs.pop('num_workers')
    )
    channels, width, height = data_module.dims
    assert width == height, f"{width=}, {height=} image not square"
    params['diffusion_kwargs']['num_classes'] = params['unet_kwargs']['num_classes'] = data_module.num_classes
    params['diffusion_kwargs']['img_size'] = params['unet_kwargs']['img_size'] = width
    params['diffusion_kwargs']['color_channels'] = params['unet_kwargs']['out_channels'] = params['unet_kwargs']['in_channels'] = channels
    
    combined_model = DiffusionWithModel(params).to(device)

    loss_precesion = 3
    loss_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=3, filename='loss-{epoch}-{val_loss:.%sf}' % loss_precesion)
    epoch_callback = ModelCheckpoint(every_n_epochs=2, filename='epoch-{epoch}-{val_loss:.%sf}' % loss_precesion)
    last_callback = ModelCheckpoint(save_last=True, filename='last-{epoch}-{val_loss:.%sf}' % loss_precesion)
    
    trainer = pl.Trainer(
        default_root_dir=models_dir,
        callbacks=[loss_callback, epoch_callback, last_callback],
        max_epochs=params.get('epochs'),
        logger=CSVLogger(models_dir, flush_logs_every_n_steps=100),
        log_every_n_steps=10,
        **hardware_kwargs
    )
    
    trainer.fit(
        combined_model,
        data_module
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware')
    parser.add_argument('--exp_name')
    
    args = parser.parse_args()
    
    hardware = 'local' if args.hardware is None else args.hardware
    exp_name = 'unnamed' if args.exp_name is None else args.exp_name
    
    params = diffusion_uncond_simple()
    
    main(
        params,
        42,
        hardware_kwargs=get_hardware_kwargs(hardware),
        experiment_name=exp_name
    )
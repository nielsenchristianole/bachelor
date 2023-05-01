import os
import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from diffusion.script_util import get_vae_mnist_kwargs
from diffusion.lightning_modules import MNISTDataModule
from diffusion.vae import SimpleVAE

def main(
    epochs: int,
    seed: int,
    *,
    hardware_kwargs: dict,
):
    work_dir = hardware_kwargs.pop('work_dir')
    data_dir = os.path.join(work_dir, 'datasets')
    models_dir = os.path.join(work_dir, 'models', f'vae_mnist-{seed}')
    
    device = torch.device(hardware_kwargs.pop('device_name'))
    
    pl.seed_everything(seed, workers=True)
    
    data_module = MNISTDataModule(
        data_dir=data_dir,
        batch_size=hardware_kwargs.pop('batch_size'),
        num_workers=hardware_kwargs.pop('num_workers'),
        normalize=False
    )
    
    vea = SimpleVAE(data_module.dims, latens_dim=2).to(device)

    loss_precesion = 3
    loss_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=3, filename='loss-{epoch}-{val_loss:.%sf}' % loss_precesion)
    
    trainer = pl.Trainer(
        default_root_dir=models_dir,
        callbacks=[loss_callback],
        max_epochs=epochs,
        logger=CSVLogger(models_dir, flush_logs_every_n_steps=100),
        log_every_n_steps=10,
        **hardware_kwargs
    )
    
    trainer.fit(
        vea,
        data_module
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware')
    
    args = parser.parse_args()
    
    hardware = 'local' if args.hardware is None else args.hardware
    seed = 43
    epochs = 20
    
    main(
        epochs,
        seed,
        hardware_kwargs=get_vae_mnist_kwargs(hardware)
    )
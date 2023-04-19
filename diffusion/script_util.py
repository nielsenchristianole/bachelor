import enum
import os
import torch

class ModelType(enum.Enum):
    """
    Which method the model uses in the reverse process
    Choices: uncond, cond_embed, guided
    """
    uncond = 'uncond' # the model generates samples blindly
    cond_embed = 'cond_embed'  # the model generates samples using embeddings
    guided = 'guided'  # the model is guided with gradients from a classifier


class VarType(enum.Enum):
    """
    How the model determines variance for sampling during the reverse process
    Choices: zero, scheduled, learned
    """
    zero = 'zero' # DDIM, the model is not deterministic
    scheduled = 'scheduled' # the model uses the beta noise schedule as variance when sampling
    learned = 'learned' # the model predicts the variance along the mean in the forward pass


def get_hardware_kwargs(hardware: str):
    if hardware == 'local':
        return dict(
            batch_size=16,
            accelerator='gpu',
            strategy='dp',
            num_nodes=1,
            num_workers=8,
            device_name='cuda',
            work_dir='C:/Users/niels/local_data/bachelor'
        )
    elif hardware == 'hpc':
        return dict(
            batch_size=16,
            accelerator='gpu',
            strategy='ddp',
            devices=torch.cuda.device_count(),
            num_nodes=1,
            num_workers=min(16, os.cpu_count()),
            device_name='cuda',
            work_dir='./'
        )
    else:
        raise NotImplementedError

def get_vgg_mnist_kwargs(hardware: str):
    if hardware == 'local':
        return dict(
            batch_size=128,
            accelerator='gpu',
            strategy='dp',
            num_nodes=1,
            num_workers=8,
            device_name='cuda',
            work_dir='C:/Users/niels/local_data/bachelor'
        )
    elif hardware == 'hpc':
        return dict(
            batch_size=128,
            accelerator='gpu',
            strategy='ddp',
            devices=torch.cuda.device_count(),
            num_nodes=1,
            num_workers=min(16, os.cpu_count()),
            device_name='cuda',
            work_dir='./'
        )
    else:
        raise NotImplementedError

def diffusion_uncond_simple() -> dict:
    unet_kwargs = dict(
        model_channels = 16,
        num_res_blocks = 8,
        dropout = 0.
    )
    diffusion_kwargs = dict(
        num_diffusion_timesteps = 100,
    )
    return dict(
        model_type = ModelType.uncond,
        var_type = VarType.scheduled,
        lr = 3e-4,
        epochs = 40,
        unet_kwargs = unet_kwargs,
        diffusion_kwargs = diffusion_kwargs
    )


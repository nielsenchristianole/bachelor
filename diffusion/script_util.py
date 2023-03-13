import enum
import os

class ModelType(enum.Enum):
    """
    Which method the model uses in the reverse process
    """
    uncond = 'uncond' # the model generates samples blindly
    cond_embed = 'cond_embed'  # the model generates samples using embeddings
    guided = 'guided'  # the model is guided with gradients from a classifier

def get_hardware_kwargs(hardware: str):
    if hardware == 'local':
        return dict(
            accelerator='gpu',
            strategy='dp',
            num_nodes=1,
            num_workers=8,
            device_name='cuda',
            work_dir='C:/Users/niels/local_data/bachelor'
        )
    elif hardware == 'hpc':
        return dict(
            accelerator='gpu',
            strategy='ddp',
            devices=8,
            num_nodes=1,
            num_workers=os.cpu_count(),
            device_name='cuda',
            work_dir='./'
        )
    else:
        raise NotImplementedError

def diffusion_uncond_defaults(size_multiplier=1, levels=3) -> dict:
    unet_kwargs = dict(
        model_channels = 8 * size_multiplier,
        num_res_blocks = 2 * size_multiplier,
        dropout = 0.,
        channel_mult = (1, 2, 4)[:levels],
        resample_mode = None,
        use_scale_shift_norm = False
    )
    diffusion_kwargs = dict(
        num_diffusion_timesteps = 50 * size_multiplier,
    )
    return dict(
        model_type = ModelType.uncond,
        lr = 3e-4,
        epochs = 8,
        batch_size = 64,
        unet_kwargs = unet_kwargs,
        diffusion_kwargs = diffusion_kwargs
    )


def diffusion_cond_embed_defaults(size_multiplier=1, levels=3) -> dict:
    unet_kwargs = dict(
        model_channels = 8 * size_multiplier,
        num_res_blocks = 2 * size_multiplier,
        dropout = 0.,
        channel_mult = (1, 2, 4)[:levels],
        resample_mode = None,
        use_scale_shift_norm = False
    )
    diffusion_kwargs = dict(
        num_diffusion_timesteps = 50 * size_multiplier,
    )
    return dict(
        model_type = ModelType.cond_embed,
        lr = 3e-4,
        epochs = 8,
        batch_size = 32,
        unet_kwargs = unet_kwargs,
        diffusion_kwargs = diffusion_kwargs
    )
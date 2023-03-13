import enum


class ModelType(enum.Enum):
    """
    Which method the model uses in the reverse process
    """
    uncond = 'uncond' # the model generates samples blindly
    cond_embed = 'cond_embed'  # the model generates samples using embeddings
    guided = 'guided'  # the model is guided with gradients from a classifier


def diffusion_uncond_defaults() -> dict:
    unet_kwargs = dict(
        model_channels = 16,
        num_res_blocks = 4,
        dropout=0.,
        channel_mult=(1, 2, 4),
        resample_mode=None,
        use_scale_shift_norm=False
    )
    diffusion_kwargs = dict(
        num_diffusion_timesteps=50,
    )
    return dict(
        model_type=ModelType.uncond,
        lr=3e-4,
        epochs=4,
        batch_size=32,
        unet_kwargs=unet_kwargs,
        diffusion_kwargs=diffusion_kwargs
    )


def diffusion_cond_embed_defaults() -> dict:
    unet_kwargs = dict(
        model_channels = 16,
        num_res_blocks = 4,
        dropout=0.,
        channel_mult=(1, 2, 4),
        resample_mode=None,
        use_scale_shift_norm=False
    )
    diffusion_kwargs = dict(
        num_diffusion_timesteps=50,
    )
    return dict(
        model_type=ModelType.cond_embed,
        lr=3e-4,
        epochs=4,
        batch_size=32,
        unet_kwargs=unet_kwargs,
        diffusion_kwargs=diffusion_kwargs
    )
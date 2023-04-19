import numpy as np
import tqdm

import torch
import torch.nn.functional as F

from .script_util import ModelType, VarType
from .vgg5 import VGG5
from .unet import SimpleUNet


class Diffusion:
    def __init__(
        self,
        num_diffusion_timesteps=200,
        img_size=64,
        color_channels=3,
        num_classes=None,
        device=None
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_size = img_size
        self.color_channels = color_channels
        self.num_classes = num_classes
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.beta = self.prepare_noise_schedule()
        self.std = np.sqrt(self.beta)
        
        self.alpha = 1. - self.beta
        self.alpha_hat = np.cumprod(self.alpha)
        
        self.sqrt_alpha_hat = np.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = np.sqrt(1 - self.alpha_hat)
        self.noise_coef = self.beta / self.sqrt_one_minus_alpha_hat
        self.recip_sqrt_alpha = 1 / np.sqrt(self.alpha)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor=None):
        """
        Sample from q(x_t | x_0) using the diffusion process
        """
        if noise is None:
            noise = torch.randn_like(x_0, device=self.device)
        
        c1 = self._get(self.sqrt_alpha_hat, t, x_0.shape)
        c2 = self._get(self.sqrt_one_minus_alpha_hat, t, x_0.shape)
        
        return c1 * x_0 + c2 * noise, noise
    
    def p_mean_std(self, model: SimpleUNet, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None):
        """
        Calculate mean and std of p(x_{t-1} | x_t) using the reverse process and model
        """
        model_out = model(x_t, t, y=y)
        if model.var_type in [VarType.zero, VarType.scheduled]:
            mu = model_out
            if model.var_type is VarType.zero:
                std = torch.zeros_like(mean)
            else:
                std = self._get(self.std, t, x_t.shape)
        elif model.var_type is VarType.learned:
            mu, var = self._split_model_out(model_out, x_t.shape)
            std = torch.sqrt(var)
        else:
            raise NotImplementedError
        
        c1 = self._get(self.recip_sqrt_alpha, t, x_t.shape)
        c2 = self._get(self.noise_coef, t, x_t.shape)
        mean = c1 * (x_t - c2 * mu)
        
        return mean, std

    def p_sample(
        self,
        model: SimpleUNet,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor=None
    ):
        """
        Sample from p(x_{t-1} | x_t) using the reverse process and model
        """
        mean, std = self.p_mean_std(model, x, t, y=y)
        noise = torch.randn_like(x, device=self.device) * (t > 0)[:, None, None, None]
        return mean + std * noise
    
    def p_sample_loop(
        self,
        model: SimpleUNet,
        t_lower: int,
        t_upper: int,
        x: torch.Tensor=None,
        y: torch.Tensor=None,
        n_samples=None,
        to_img=False,
        show_pbar=False
    ):
        """
        Loops through the reverse process from t_upper to t_lower
        """
        assert t_lower <= t_upper
        assert ((x is None and n_samples is not None) or
                (x is not None and n_samples is None))
        assert not model.training, "UNet should not be in train mode during sampling"
        
        # generetes samples if none are passed
        if n_samples:
            x = torch.randn((n_samples, self.color_channels, self.img_size, self.img_size), device=self.device)
        shape = x.shape
        
        iterable = list(reversed(range(t_lower, t_upper+1)))
        iterable = tqdm.tqdm(iterable) if show_pbar else iterable
        for i in iterable:
            t = torch.tensor([i] * shape[0], device=self.device)
            with torch.no_grad():
                x = self.p_sample(model, x, t, y=y)
        
        return self.to_img(x) if to_img else x
 
    def sample(self, model: SimpleUNet, n_samples, y: bool|torch.Tensor=None, show_pbar=True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples unconditionally using the reverse process starting from gaussian noise, returns labels
        """
        if isinstance(y, bool):
            # set to None if y==False
            y = self.sample_classes(n_samples) if y else None
        elif isinstance(y, torch.Tensor):
            assert y.shape(0) == n_samples, \
                f"{y.shape=} not campatible with {n_samples=}"
        elif y is None:
            pass
        else:
            raise NotImplementedError
        samples = self.p_sample_loop(
            model,
            t_lower=0,
            t_upper=self.num_diffusion_timesteps-1,
            y=y,
            n_samples=n_samples,
            to_img=True,
            show_pbar=show_pbar
            )
        return samples, y

    def p_guided_mean_std(
        self,
        model: SimpleUNet,
        classifier: VGG5,
        x_t: torch.Tensor,
        x_0_reconstructed: torch.Tensor,
        original_latent: torch.Tensor,
        t: torch.Tensor,
        y_target: torch.Tensor,
        *,
        lambda_p: float,
        lambda_c: float,
        vgg_block: int
    ):
        """
        Algo 1 from _Diffusion Models Beat GANs on Image Synthesis_
        """
        assert not model.training, "UNet should not be in training when doing guided diffusion"
        assert not classifier.training, "Classifier should not be in training when doing guided diffusion"
        # cloning reconstructed image to get gradient
        x_0_clone = torch.clone(x_0_reconstructed)
        x_0_clone.requires_grad_(True)
        
        # for class loss
        out_logits = classifier.forward(x_0_clone)
        class_loss = F.cross_entropy(out_logits, y_target, reduction='none')
        
        # for perception loss
        reconstructed_latent = classifier.get_featuremap(x_0_clone, vgg_block)
        perc_loss = F.mse_loss(reconstructed_latent, original_latent, reduction='none').mean(dim=(1,2,3))
        
        # combined loss and getting grad
        reconstructed_loss = lambda_c * class_loss + lambda_p * perc_loss
        noisy_loss = self._get(self.recip_sqrt_alpha, t, reconstructed_loss.shape) * reconstructed_loss # self.recip_sqrt_alpha[t.cpu().numpy()] * reconstructed_loss
        noisy_loss.mean().backward()
        gradient = x_0_clone.grad.detach()
        
        with torch.no_grad():
            model_out = model(x_t, t)
            
            if model.var_type in [VarType.zero, VarType.scheduled]:
                mu = model_out
                cov = self._get(self.beta, t, x_t.shape)
            
                if model.var_type is VarType.zero:
                    std = torch.zeros_like(mean)
                else:
                    std = self._get(self.std, t, x_t.shape)
            elif model.var_type is VarType.learned:
                mu, cov = self._split_model_out(model_out, x_t.shape)
                std = torch.sqrt(cov)
            else:
                raise NotImplementedError(f"{model.var_type=} not implemented")
            
            c1 = self._get(self.recip_sqrt_alpha, t, x_t.shape)
            c2 = self._get(self.noise_coef, t, x_t.shape)
            mean = c1 * (x_t - c2 * mu) - cov * gradient
            
            return mean, std
    
    def p_guided_sample(
        self,
        model: SimpleUNet,
        classifier: VGG5,
        x_t: torch.Tensor,
        x_0_reconstructed: torch.Tensor,
        original_latent: torch.Tensor,
        t: torch.Tensor,
        y_target: torch.Tensor,
        *,
        lambda_p: float,
        lambda_c: float,
        vgg_block: int
    ):
        """
        Sample from p(x_{t-1} | x_t) using the guided reverse process
        """
        mean, std = self.p_guided_mean_std(
            model,
            classifier,
            x_t,
            x_0_reconstructed,
            original_latent,
            t,
            y_target,
            lambda_p=lambda_p,
            lambda_c=lambda_c,
            vgg_block=vgg_block
        )
        noise = torch.randn_like(x_t, device=self.device) * (t > 0)[:, None, None, None]
        return mean + std * noise
        

    def guided_counterfactual(
        self,
        model: SimpleUNet,
        classifier: VGG5,
        x_0: torch.Tensor,
        y: int,
        tau: int,
        *,
        lambda_p: float,
        lambda_c: float,
        vgg_block: int,
        show_pbar: bool=True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Uses the algoritm proposed in _Diffusion Models for Counterfactual Explanations_ for generating
        a counterfactual explenaition.
        Returns the counterfactual, the classification logits, and weather or not it has fooled the classifier
        remember tau is 0th indexed. Wanting to do a single guided reverse step should be tau=0
        """
        assert tau < (T:=self.num_diffusion_timesteps), \
            f"{tau=} has to be smaller than {T=}"
        
        batch_size = x_0.shape[0]
        
        x_0_original = x_0.to(self.device)
        y = torch.tensor((y,) * batch_size, device=self.device)
        t = torch.tensor((tau,) * batch_size, device=self.device)
        
        x_t, noise = self.q_sample(x_0_original, t)
        x_0_reconstructed = torch.clone(x_0_original)
        
        with torch.no_grad():
            original_latent = classifier.get_featuremap(x_0_original, vgg_block)
        
        iterator = list(reversed(range(0, tau)))
        iterator = tqdm.tqdm(iterator) if show_pbar else iterator
        
        x_t = self.p_guided_sample(model, classifier, x_t, x_0_reconstructed, original_latent, t, y, lambda_p=lambda_p, lambda_c=lambda_c, vgg_block=vgg_block)
        for i in iterator:
            t = torch.tensor((i,), device=self.device)
            x_0_reconstructed = self.p_sample_loop(model, t_lower=0, t_upper=i, x=x_t)
            x_t = self.p_guided_sample(model, classifier, x_t, x_0_reconstructed, original_latent, t, y, lambda_p=lambda_p, lambda_c=lambda_c, vgg_block=vgg_block)
        classification_logits = classifier.forward(x_t)
        
        return x_t, classification_logits
    
    def sample_timesteps(self, n_samples):
        """
        Samples timesteps uniformly for the training
        """
        return torch.randint(low=0, high=self.num_diffusion_timesteps, size=(n_samples,), device=self.device)
    
    def sample_classes(self, n_samples):
        """
        Samples classes uniformly
        """
        return torch.randint(low=0, high=self.num_classes, size=(n_samples,), device=self.device)
    
    def prepare_noise_schedule(self):
        """
        Creates a noise schedule with a given number of diffusion steps
        """
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, self.num_diffusion_timesteps)
        assert (betas > 0).all() and (betas <= 1).all(), f"{self.num_diffusion_timesteps=} incompateble, 0 < beta_s <= 1"
        return betas
    
    @staticmethod
    def _split_model_out(
        model_out: torch.Tensor,
        in_shape: torch.Size
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C = in_shape[:2]
        assert model_out.shape == (B, C * 2, *in_shape[2:]), "out channel dim not twice input channel dim, cannot split to mean, var"
        return torch.split(model_out, C, dim=1)
    
    @staticmethod
    def to_img(x: torch.Tensor):
        """
        Takes a tensor and converts it to rgb
        """
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
    @staticmethod
    def _get(arr: np.ndarray, t: torch.IntTensor, to_shape: torch.Size):
        """
        Extracts value from initialized array to tensor
        """
        res = torch.from_numpy(arr).to(t.device)[t].float()
        while len(res.shape) < len(to_shape):
            res = res[..., None]
        return res.expand(to_shape)
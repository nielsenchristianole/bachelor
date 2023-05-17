from collections.abc import Sequence
import numpy as np
import tqdm

import torch
import torch.nn.functional as F

from .script_util import ModelType, VarType
from .vgg5 import VGG5
from .unet import SimpleUNet

# annaying workaround
Sequence.register(np.ndarray)

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

        # betas
        self.beta = self.prepare_noise_schedule()
        self.std = np.sqrt(self.beta)
        
        # alphas
        self.alpha = 1. - self.beta
        self.alpha_bar = np.cumprod(self.alpha)
        self.alpha_bar_t_minus_one = np.append(1., self.alpha_bar[:-1]) # shift right
        
        # sqrt alphas
        self.sqrt_alpha_bar = np.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = np.sqrt(1. - self.alpha_bar)
        self.recip_sqrt_alpha = 1. / np.sqrt(self.alpha)
        
        # extra betas
        self.beta_tilde = (1. - self.alpha_bar_t_minus_one) / (1. - self.alpha_bar) * self.beta
        self.log_beta = np.log(self.beta)
        self.log_beta_tilde = np.log(np.append(self.beta_tilde[1], self.beta_tilde[1:])) # first element not -inf
        
        # misc.
        self.noise_coef = self.beta / self.sqrt_one_minus_alpha_bar
        self.posterior_variance = self.beta * (1. - self.alpha_bar_t_minus_one) / (1. - self.alpha_bar)
        self.posterior_log_variance = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:])) # first element not -inf
        self.posterior_mean_coef1 = self.beta * np.sqrt(self.alpha_bar_t_minus_one) / (1.0 - self.alpha_bar)
        self.posterior_mean_coef2 = (1.0 - self.alpha_bar_t_minus_one) * np.sqrt(self.alpha) / (1.0 - self.alpha_bar)
        self.sqrt_one_minus_beta = np.sqrt(1 - self.beta)
        
    def forward_step(self, x_t: torch.Tensor, t: torch.IntTensor, noise: torch.Tensor=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_t, device=self.device)

        std = self._get(self.std, t, x_t.shape)
        sqrt_one_minus_beta = self._get(self.sqrt_one_minus_beta, t, x_t.shape)
        
        return sqrt_one_minus_beta * x_t + std * noise
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor=None):
        """
        Sample from q(x_t | x_0) using the diffusion process
        """
        if noise is None:
            noise = torch.randn_like(x_0, device=self.device)
        
        c1 = self._get(self.sqrt_alpha_bar, t, x_0.shape)
        c2 = self._get(self.sqrt_one_minus_alpha_bar, t, x_0.shape)
        
        return c1 * x_0 + c2 * noise, noise
    
    def model_err_pred_to_mean(self, err: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Used to calculate the estimate for the mean in the reverse process using the predicted noise
        """
        c1 = self._get(self.recip_sqrt_alpha, t, x_t.shape)
        c2 = self._get(self.noise_coef, t, x_t.shape)
        return c1 * (x_t - c2 * err)
    
    def model_v_pred_to_std(self, v: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Used to calculate the estimate for the variance in the reverse process using the predicted noise, only for learned variance
        """
        out = dict()
        # v in [-1, 1] -> v in [0, 1]
        v = (v + 1) / 2
        log_beta = self._get(self.log_beta, t, v.shape)
        log_beta_tilde = self._get(self.log_beta_tilde, t, v.shape)
        log_var = v * log_beta + (1 - v) * log_beta_tilde
        out['std'] = torch.exp(0.5 * log_var) # look how good I am at rules of exponent calculations B)
        out['log_var'] = log_var
        return out
    
    def p_mean_std(self, model: SimpleUNet, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None) -> dict[str, torch.Tensor]:
        """
        Calculate mean and std of p(x_{t-1} | x_t) using the reverse process and model
        """
        out = dict()
        model_out = model(x_t, t, y=y)
        if model.var_type in [VarType.zero, VarType.scheduled]:
            err = model_out
            if model.var_type is VarType.zero:
                out['std'] = torch.zeros_like(x_t)
            else:
                out['std'] = self._get(self.std, t, x_t.shape)
        elif model.var_type is VarType.learned:
            err, v = self._split_model_out(model_out, x_t.shape)
            out |= self.model_v_pred_to_std(v, t) # add to dict
        else:
            raise NotImplementedError
        
        out['mean'] = self.model_err_pred_to_mean(err, x_t, t)
        return out

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
        out = self.p_mean_std(model, x, t, y=y)
        mean, std = out['mean'], out['std']
        # set noise to zero when doing the last step
        noise = torch.randn_like(x, device=self.device) * (t > 0)[:, None, None, None]
        return mean + std * noise
    
    def p_sample_loop(
        self,
        model: SimpleUNet,
        t_lower: int,
        t_upper: int,
        x: torch.Tensor=None,
        y: torch.Tensor=None,
        n_samples: int=None,
        to_img: bool=False,
        show_pbar: bool=False
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
            assert t_upper == self.num_diffusion_timesteps-1
            x = torch.randn((n_samples, self.color_channels, self.img_size, self.img_size), device=self.device)
        shape = x.shape
        
        iterable = list(reversed(range(t_lower, t_upper+1)))
        iterable = tqdm.tqdm(iterable) if show_pbar else iterable
        for i in iterable:
            t = torch.tensor([i] * shape[0], device=self.device)
            with torch.no_grad():
                x = self.p_sample(model, x, t, y=y)
        
        return self.to_img(x) if to_img else x
 
    def sample(
        self,
        model: SimpleUNet,
        n_samples: int=1,
        y: bool|torch.Tensor=None,
        show_pbar=True,
        to_img=True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples using the reverse process starting from gaussian noise, returns labels
        """
        if isinstance(y, bool):
            y = self.sample_classes(n_samples) if y else None # set to None if y==False
        elif (y is None) or isinstance(y, torch.Tensor):
            pass
        elif isinstance(y, Sequence):
            y = torch.Tensor(y).to(self.device).long()
        else:
            raise NotImplementedError
        assert (y is None) or (y.shape[0] == n_samples), f"{y.shape=} not campatible with {n_samples=}"
        samples = self.p_sample_loop(
            model,
            t_lower=0,
            t_upper=self.num_diffusion_timesteps-1,
            y=y,
            n_samples=n_samples,
            to_img=to_img,
            show_pbar=show_pbar
            )
        return samples, y

    def get_guiding_gradient(
        self,
        classifier: VGG5,
        x_0_reconstructed: torch.Tensor,
        original_latent: torch.Tensor,
        t: torch.Tensor,
        y_target: torch.Tensor,
        lambda_c: float,
        lambda_p: float,
        vgg_block: int,
    ):
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
        return gradient

    def p_guided_mean_std(
        self,
        model: SimpleUNet,
        classifier: VGG5,
        x_t: torch.Tensor,
        x_0_reconstructed: torch.Tensor|list[torch.Tensor],
        original_latent: torch.Tensor,
        t: torch.Tensor,
        y_target: torch.Tensor,
        *,
        lambda_p: float,
        lambda_c: float,
        vgg_block: int
    ):
        """
        Algo 1 from _Diffusion Models Beat GANs on Image Synthesis
        Calculates the mean and std for a single reverse step guided with a classifier
        """
        assert not model.training, "UNet should not be in training when doing guided diffusion"
        assert not classifier.training, "Classifier should not be in training when doing guided diffusion"
        
        if isinstance(x_0_reconstructed, list):
            gradients = list()
            for tmp_reconstructed in x_0_reconstructed:
                gradients.append(
                    self.get_guiding_gradient(
                        classifier=classifier,
                        x_0_reconstructed=tmp_reconstructed,
                        original_latent=original_latent,
                        t=t,
                        y_target=y_target,
                        lambda_c=lambda_c,
                        lambda_p=lambda_p,
                        vgg_block=vgg_block
                    )[None, ...]
                )
            gradient = torch.cat(gradients).mean(dim=0)
        else:
            gradient = self.get_guiding_gradient(
                classifier=classifier,
                x_0_reconstructed=x_0_reconstructed,
                original_latent=original_latent,
                t=t,
                y_target=y_target,
                lambda_c=lambda_c,
                lambda_p=lambda_p,
                vgg_block=vgg_block
            )
        
        with torch.no_grad():
            model_out = model(x_t, t)
            
            if model.var_type in [VarType.zero, VarType.scheduled]:
                err = model_out
                cov = self._get(self.beta, t, x_t.shape)
            
                if model.var_type is VarType.zero:
                    std = torch.zeros_like(mean)
                else:
                    std = self._get(self.std, t, x_t.shape)
            elif model.var_type is VarType.learned:
                err, v = self._split_model_out(model_out, x_t.shape)
                v_out = self.model_v_pred_to_std(v, t)
                std = v_out['std']
                cov = torch.exp(v_out['log_var'])
            else:
                raise NotImplementedError(f"{model.var_type=} not implemented")
            
            mean = self.model_err_pred_to_mean(err, x_t, t) - cov * gradient
            
            return mean, std
    
    def p_guided_sample(
        self,
        model: SimpleUNet,
        classifier: VGG5,
        x_t: torch.Tensor,
        x_0_reconstructed: torch.Tensor|list[torch.Tensor],
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

    def guided_sample_iter(
        self,
        model: SimpleUNet,
        classifier: VGG5,
        x_t: torch.Tensor,
        t: int,
        y_target: torch.Tensor,
        *,
        x_0_original: torch.Tensor=None,
        x_0_latent: torch.Tensor=None,
        n_reconstruct_samples: int=1,
        lambda_p: float,
        lambda_c: float,
        vgg_block: int,
        return_reconstruct: bool=False
    ) -> torch.Tensor|tuple[torch.Tensor, torch.Tensor|list[torch.Tensor]]:
        """
        From a noisy image x_t, sample uncon to get clean image, and do a reverse step
        """
        assert (x_0_original is not None) or (x_0_latent is not None)
        
        if x_0_latent is None:
            with torch.no_grad():
                x_0_latent = classifier.get_featuremap(x_0_original, vgg_block)
        
        x_0_reconstructed = self.p_sample_loop(model, t_lower=0, t_upper=t, x=x_t)
        if n_reconstruct_samples > 1:
            x_0_reconstructed = [x_0_reconstructed]
            for _ in range(n_reconstruct_samples-1):
                x_0_reconstructed.append(
                    self.p_sample_loop(model, t_lower=0, t_upper=t, x=x_t)
                )
        t_tensor = torch.tensor([t] * x_t.shape[0], device=self.device)
        x_t = self.p_guided_sample(
            model,
            classifier,
            x_t,
            x_0_reconstructed,
            x_0_latent,
            t_tensor,
            y_target,
            lambda_p=lambda_p,
            lambda_c=lambda_c,
            vgg_block=vgg_block
        )
        if return_reconstruct:
            return x_t, x_0_reconstructed
        return x_t


    def guided_counterfactual(
        self,
        model: SimpleUNet,
        classifier: VGG5,
        x_0: torch.Tensor,
        y: int,
        tau: int,
        *,
        n_reconstruct_samples: int=1,
        lambda_p: float,
        lambda_c: float,
        vgg_block: int,
        show_pbar: bool=True,
        return_classifier_logits: bool=False,
        return_pred_class: bool=False,
        return_intermediate: bool=False,
    ) -> torch.Tensor|tuple[torch.Tensor, ...]:
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
        
        x_t, _ = self.q_sample(x_0_original, t)
        x_0_reconstructed = torch.clone(x_0_original)
        
        with torch.no_grad():
            original_latent = classifier.get_featuremap(x_0_original, vgg_block)
        
        iterator = list(reversed(range(0, tau)))
        iterator = tqdm.tqdm(iterator) if show_pbar else iterator
        
        if return_intermediate:            
            assert (batch_size == 1) or (n_reconstruct_samples == 1), f'return_intermediate not defined when neither {batch_size=} nor {n_reconstruct_samples=} equals 1'
            intermediate_reconstructed = [x_0_reconstructed.repeat((n_reconstruct_samples, 1, 1, 1))[None, ...]]

        x_t = self.p_guided_sample(
            model,
            classifier,
            x_t,
            x_0_reconstructed,
            original_latent,
            t,
            y,
            lambda_p=lambda_p,
            lambda_c=lambda_c,
            vgg_block=vgg_block
        )
        for i in iterator:
            x_t, x_0_reconstructed = self.guided_sample_iter(
                model,
                classifier,
                x_t,
                i,
                y,
                x_0_latent=original_latent,
                n_reconstruct_samples=n_reconstruct_samples,
                lambda_p=lambda_p,
                lambda_c=lambda_c,
                vgg_block=vgg_block,
                return_reconstruct=True
            )
            if return_intermediate:
                if isinstance(x_0_reconstructed, list):
                    x_0_reconstructed = torch.cat(x_0_reconstructed)
                intermediate_reconstructed.append(x_0_reconstructed[None, ...])
        
        outputs = (x_t,)
        
        classification_logits = None
        if return_classifier_logits:
            classification_logits = classifier.forward(x_t)
            outputs += (classification_logits,)
        if return_pred_class:
            classification_logits = classification_logits or classifier.forward(x_t)
            pred_class = torch.argmax(classification_logits, dim=1)
            outputs += (pred_class,)
        if return_intermediate:
            outputs += (torch.cat(intermediate_reconstructed),)
        
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
    
    def sample_timesteps(self, n_samples):
        """
        Samples timesteps uniformly for the training
        """
        return torch.randint(low=0, high=self.num_diffusion_timesteps, size=(n_samples,), device=self.device)
    
    def sample_classes(self, n_samples):
        """
        Samples classes uniformly
        """
        return torch.randint(low=0, high=self.num_classes, size=(n_samples,), device=self.device).long()
    
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
    
    def calculate_loss_vlb(
        self,
        err: torch.Tensor,
        v: torch.Tensor,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate and returns the variational lower bound loss
        """
        true_mean = self._get(self.posterior_mean_coef1, t, x_t.shape) * x_0 + self._get(self.posterior_mean_coef2, t, x_t.shape) * x_t
        true_log_var = self._get(self.posterior_log_variance, t, x_t.shape)
        pred_mean = self.model_err_pred_to_mean(err, x_t, t)
        pred_log_var = self.model_v_pred_to_std(v, t)['log_var']
        
        # loss term from Diffusion Models Beat GANs on Image Synthesis
        loss_vlb = 0.5 * (
        -1.0
        + pred_log_var
        - true_log_var
        + torch.exp(true_log_var - pred_log_var)
        + ((true_mean - pred_mean) ** 2) * torch.exp(-pred_log_var)
        )
        # take mean over non batch dim
        loss_vlb = loss_vlb.mean(dim=list(range(1, len(loss_vlb.shape))))
        # set loss to 0 when t is 0|T. No loss when no variance or when x_T is pure noise
        # last does not happen during training as the loss is only calculated for the reverse process
        return torch.where((t == 0), torch.zeros_like(loss_vlb), loss_vlb).mean()
    
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
        x = x.clamp(0., 1.)
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
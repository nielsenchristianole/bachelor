import torch
import tqdm
from diffusion.evaluations import calculate_FVAED

import os

import numpy as np
from numpy import random
import pandas as pd
import glob
import math
import tqdm
import tabulate

import torch
import matplotlib.pyplot as plt

from diffusion.data_util import get_ckpt_path, show_out_images, get_logger_path, train_plot, img_plot
from diffusion.script_util import ModelType, VarType
from diffusion.lightning_modules import DiffusionWithModel, MNISTDataModule
from diffusion.vgg5 import VGG5
from diffusion.vae import SimpleVAE

"""
This script is to generate the fid vs timestep plot
"""

number_of_plot_timesteps = 20
batch_size = 64
stage = 'fit' # fit|test
num_labels = 10

evaluation_result_dir = './fid_vs_timestep_results/'
work_dir = r'C:\Users\niels\local_data\bachelor'

data_dir = os.path.join(work_dir, 'datasets')
models_dir = os.path.join(work_dir, 'models')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = os.path.join(work_dir, 'eval_models', 'uncond_ddpm.ckpt')
classifier_path = os.path.join(work_dir, 'eval_models', 'classifier.ckpt')
vae_path = os.path.join(work_dir, 'eval_models', 'vae.ckpt')
classifier = VGG5.load_from_checkpoint(classifier_path).to(device).eval()
vae = SimpleVAE.load_from_checkpoint(vae_path).to(device).eval()

combined = DiffusionWithModel.load_from_checkpoint(model_path).to(device)
model, diffusion = combined.extract_models()

data_module = MNISTDataModule(data_dir, batch_size, num_workers=0)

list_FVAED = [list() for _ in range(num_labels)]
list_accuracy = [list() for _ in range(num_labels)]
list_perception_diff = [list() for _ in range(num_labels)]

timesteps = np.linspace(0, diffusion.num_diffusion_timesteps-1, number_of_plot_timesteps, endpoint=True, dtype=np.int64)
# timesteps = [0, 1, 2]

with torch.no_grad():
    for label in range(num_labels):
        print(f'label: {label}', flush=True)
        
        data_module.label_subset = [label]
        data_module.setup(stage)
        tmp_dataloader = data_module.val_dataloader()
        
        all_x_true = list()
        all_featuremaps_true = list()
        for batch in tqdm.tqdm(iter(tmp_dataloader), desc='encoding val set', total=len(tmp_dataloader)):
            x, _ = batch
            x = x.to(device)
            encoded_x_true = vae.get_latent(x)
            all_x_true.append(encoded_x_true)
            all_featuremaps_true.append(x.detach().cpu().numpy())

        all_x_true = torch.cat(all_x_true)
        x_true_mean = all_x_true.mean(dim=0)
        x_true_cov = torch.cov(all_x_true.T)

        for timestep in tqdm.tqdm(timesteps, desc='encoding reverse process'):
            all_x_0 = list()
            all_x_0_labels = list()
            all_featuremap_distances = list()
            for batch_idx, batch in enumerate(iter(tmp_dataloader)):
                x, y = batch
                x = x.to(device)
                t = torch.tensor((timestep,)*len(y), device=device)
                x_t, _ = diffusion.q_sample(x, t)
                x_0 = diffusion.p_sample_loop(model, 0, timestep, x_t)
                x_0 = x_0.clamp(0, 1)
                encoded_x_0 = vae.get_latent(x_0)
                all_x_0.append(encoded_x_0)
                predicted_label = classifier(x_0).argmax(dim=1)
                all_x_0_labels.append(predicted_label)
                distance = np.square(all_featuremaps_true[batch_idx] - x_0.detach().cpu().numpy()).sum(axis=(1,2,3))
                all_featuremap_distances.append(distance)
                
                
            all_x_0 = torch.cat(all_x_0)
            all_x_0_labels = torch.cat(all_x_0_labels).cpu().numpy()
            
            x_0_mean = all_x_0.mean(dim=0)
            x_0_cov = torch.cov(all_x_0.T)

            timestep_accuracy = (all_x_0_labels == label).mean()
            list_accuracy[label].append(timestep_accuracy)
            
            timestep_fvaed = calculate_FVAED(x_0_mean, x_0_cov, x_true_mean, x_true_cov)
            list_FVAED[label].append(timestep_fvaed)
            
            distance = np.concatenate(all_featuremap_distances).mean()
            list_perception_diff[label].append(distance)


# create directory if not exists
os.makedirs(evaluation_result_dir, exist_ok=True)
np.save(evaluation_result_dir + 'fvaed.npy', list_FVAED)
np.save(evaluation_result_dir + 'timesteps.npy', timesteps)
np.save(evaluation_result_dir + 'prediction_accuracy.npy', list_accuracy)
np.save(evaluation_result_dir + 'pixel_distance.npy', list_perception_diff)
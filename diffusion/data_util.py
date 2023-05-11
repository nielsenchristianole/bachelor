import os
import math

import numpy as np
import pandas as pd

import torch

import matplotlib
import matplotlib.pyplot as plt

from glob import glob
from functools import partial

from .script_util import ModelType


def get_ckpt_path_long(model_type: ModelType=None, work_dir='./', version_suffix=-1, mode='last', epoch=0, pattern=''):
    start_path = os.path.join(work_dir, '**/*.ckpt')
    all_paths = glob(start_path, recursive=True)
    if model_type is not None:
        all_paths = [p for p in all_paths if model_type.name in p]
    
    # get version
    if version_suffix == -1:
        vs = []
        for path in all_paths:
            v = [v for v in path.split('\\') if 'version' in v][0]
            vs.append(int(v.split('_')[-1]))
        version_suffix = max(vs)
    version = f'version_{version_suffix}'
    
    if mode == 'last':
        return [p for p in all_paths if ((version in p) and ('last-' in p))][0]
    elif mode == 'best':
        scores = []
        val_pattern = 'val_loss='
        val_paths = [p for p in all_paths if val_pattern in p]
        for path in val_paths:
            tmp = path.split(val_pattern)[-1][:-5]
            if 'v' not in tmp:
                scores.append(float(path.split(val_pattern)[-1][:-5]))
        score = f'{val_pattern}{max(scores)}'
        return [p for p in val_paths if ((version in p) and (score in p))][0]
    elif mode == 'epoch':
        return [p for p in all_paths if ((version in p) and (f'epoch={epoch}' in p))][0]
    elif mode == 'match':
        return [p for p in all_paths if ((version in p) and (pattern in p))][0]
    else:
        raise NotImplementedError


def get_logger_path_long(work_dir, version_suffix=-1):
    start_path = os.path.join(work_dir, '**/*.csv')
    all_paths = glob(start_path, recursive=True)
    
    # get version
    if version_suffix == -1:
        vs = []
        for path in all_paths:
            v = [v for v in path.split('\\') if 'version' in v][0]
            vs.append(int(v.split('_')[-1]))
        version_suffix = max(vs)
    version = f'version_{version_suffix}'
    paths = [p for p in all_paths if version in p]
    assert len(paths) == 1, f"{paths=} paths are ambigious"
    return paths[0]


work_dir='C:/Users/niels/local_data/bachelor/models'
get_ckpt_path = partial(get_ckpt_path_long, work_dir=work_dir)
get_logger_path = partial(get_logger_path_long, work_dir=work_dir)


def img_plot(img, title=None, ax: matplotlib.axes.Axes=None, **ax_kwargs) -> matplotlib.axes.Axes:
    kwargs = dict(
        interpolation='nearest'
    ) | ax_kwargs
    if ax is None:
        ax = plt.gca()
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    ax.imshow(img, **kwargs)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    return ax
    

def show_out_images(x: torch.Tensor, y: torch.Tensor=None, fig=None, ax_kwargs={}, **fig_kwargs):
    x = x.detach().to('cpu').numpy()
    n_batches, channels, width, height = x.shape
    
    y = y.detach().to('cpu').numpy() if isinstance(y, torch.Tensor) else y
    y = (None for _ in range(n_batches)) if y is None else y
    
    if channels == 1:
        imgs = np.squeeze(x, 1)
    elif channels == 3:
        imgs = np.moveaxis(x, 1, -1)
    else:
        raise NotImplementedError
    
    num_side_img = math.ceil(math.sqrt(n_batches))
    if fig == None:
        fig_kwargs['figsize'] = fig_kwargs.get('figsize', (2*num_side_img, 2*num_side_img))
        fig = plt.figure(**fig_kwargs)
    for i, (img, label) in enumerate(zip(imgs, y), start=1):
        ax = fig.add_subplot(num_side_img, num_side_img, i)
        img_plot(img, label, ax, **ax_kwargs)


def training_parrent_plot(
    x: str,
    y: str,
    x_label: str,
    y_label:str,
    plot_title: str,
    df: pd.DataFrame,
    ax: matplotlib.axes.Axes=None,
    yscale='linear',
    window=20,
    legend=True,
    lines_kwargs={},
    scatter_kwargs={}
):
    df_train = df[[x, y]].dropna().set_index(x)
    df_train['rolled'] = df_train[y].rolling(window=window).mean()
    if ax is None:
        ax = plt.gca()
    
    # basics
    ax.set_yscale(yscale)
    ax.set_title(plot_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # rolling average plot
    kwargs = dict(
        c='C1',
        label=f'Rolled',
    ) | scatter_kwargs
    ax.plot(df_train['rolled'], **kwargs)
    
    # scatter plot
    kwargs = dict(
        s=5, # marker size
        c='C0',
        label='10 step loss avg',
    ) | scatter_kwargs
    ax.scatter(df_train.index, df_train['train_loss'], **kwargs)
    if legend:
        ax.legend()


train_plot = partial(training_parrent_plot, 'step', 'train_loss', 'train step', 'MSELoss', 'Train step loss')
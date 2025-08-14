
import logging
import glob
from types import new_class

import random
import numpy as np
import h5py
import math
import matplotlib
import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset
from mindspore.nn import RMSELoss
from mindspore import dtype as mstype
from mindspore.ops import operations as P


class PeriodicPad2d(nn.Cell):
    """ 
        pad longitudinal (left-right) circular 
        and pad latitude (top-bottom) with zeros
    """
    def __init__(self, pad_width):
       super(PeriodicPad2d, self).__init__()
       self.pad_width = pad_width

    def construct(self, x):
        out = ops.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular") 
        out = ops.pad(out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0) 
        return out

def reshape_fields(img, inp_or_tar, params, train, normalize=True, orog=None, add_noise=False):

    if len(np.shape(img)) == 3:
        img = np.expand_dims(img, 0)
    
    if np.shape(img)[2] == 721:
        img = img[:,:, 0:720, :] # remove last pixel

    n_history = np.shape(img)[0] - 1
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]
    n_channels = np.shape(img)[1] # this will either be N_in_channels or N_out_channels
    channels = params.in_channels if inp_or_tar =='inp' else params.out_channels

    if normalize and params.normalization == 'minmax':
        maxs = np.load(params.global_maxs_path)[:, channels]
        mins = np.load(params.global_mins_path)[:, channels]
        img = (img - mins) / (maxs - mins)

    if normalize and params.normalization == 'zscore':
        means = np.load(params.global_means_path)[:, channels]
        stds = np.load(params.global_stds_path)[:, channels]
        img -=means
        img /=stds

    if normalize and params.normalization == 'zscore_lat':
        means = np.load(params.global_lat_means_path)[:, channels,:720]
        stds = np.load(params.global_lat_stds_path)[:, channels,:720]
        img -=means
        img /=stds

    if params.add_grid:
        if inp_or_tar == 'inp' and params.gridtype == 'linear':
            assert params.N_grid_channels == 2, "N_grid_channels must be set to 2 for gridtype linear"
            x = np.meshgrid(np.linspace(-1, 1, img_shape_x))
            y = np.meshgrid(np.linspace(-1, 1, img_shape_y))
            grid_x, grid_y = np.meshgrid(y, x)
            grid = np.stack((grid_x, grid_y), axis = 0)
        if inp_or_tar == 'inp' and params.gridtype == 'sinusoidal':
            assert params.N_grid_channels == 4, "N_grid_channels must be set to 4 for gridtype sinusoidal"
            x1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_x)))
            x2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_x)))
            y1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_y)))
            y2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_y)))
            grid_x1, grid_y1 = np.meshgrid(y1, x1)
            grid_x2, grid_y2 = np.meshgrid(y2, x2)
            grid = np.expand_dims(np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis = 0), axis = 0)
        img = np.concatenate((img, grid), axis = 1 )

    if params.orography and inp_or_tar == 'inp':
        orog = np.expand_dims(orog, axis = (0,1))
        orog = np.repeat(orog, repeats=img.shape[0], axis=0)
        img = np.concatenate((img, orog), axis = 1)
        n_channels += 1

    img = np.squeeze(img)

    if add_noise:
        img = img + np.random.normal(0, scale=params.noise_std, size=img.shape)

    return Tensor.from_numpy(img).astype(mstype.float32)
          
def reshape_finetune_fields(img, inp_or_tar, params, normalize=True):

    if len(np.shape(img)) == 3:
        img = np.expand_dims(img, 0)
    
    if np.shape(img)[2] == 361:
        img = img[:,:, 0:360, :]
    
    n_history   = np.shape(img)[0] - 1
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]

    if inp_or_tar =='inp':
        channels = params.finetune_in_channels
    if inp_or_tar == 'force':
        channels = params.finetune_force_channels
    if inp_or_tar == 'tar':
        channels = params.finetune_out_channels

    if normalize and params.normalization == 'minmax':
        maxs = np.load(params.finetune_global_maxs_path)[:, channels]
        mins = np.load(params.finetune_global_mins_path)[:, channels]
        img = (img - mins) / (maxs - mins)

    if normalize and params.normalization == 'zscore':
        means = np.load(params.finetune_global_means_path)[:, channels]
        stds  = np.load(params.finetune_global_stds_path)[:, channels]
        img -= means
        img /= stds

    img = np.squeeze(img)

    return img

def vis_precip(fields):
    pred, tar = fields
    fig, ax = plt.subplots(1, 2, figsize=(24,12))
    ax[0].imshow(pred, cmap="coolwarm")
    ax[0].set_title("tp pred")
    ax[1].imshow(tar, cmap="coolwarm")
    ax[1].set_title("tp tar")
    fig.tight_layout()
    return fig

def read_max_min_value(min_max_val_file_path):
    with h5py.File(min_max_val_file_path, 'r') as f:
        max_values = f['max_values']
        min_values = f['min_values']
    return max_values, min_values
    
    




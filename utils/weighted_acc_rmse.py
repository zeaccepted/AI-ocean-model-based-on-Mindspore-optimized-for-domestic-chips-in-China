

import os
import time
import numpy as np
import argparse
import h5py
from icecream import ic
from collections import OrderedDict
from utils import logging_utils
logging_utils.config_logger()

import wandb

import warnings
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds



def unlog_tp(x, eps=1E-5):
    return eps*(np.exp(x)-1)


def mean(x, axis = None):
    y = np.sum(x, axis) / np.size(x, axis)
    return y

def lat_np(j, num_lat):
    return 90. - j * 180./(num_lat-1)


def top_quantiles_error(pred, target):
    if len(pred.shape) ==2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) ==2:
        target = np.expand_dims(target, 0)
    qs = 100
    qlim = 5
    qcut = 0.1
    qtile = 1. - np.logspace(-qlim, -qcut, num=qs)
    P_tar = np.quantile(target, q=qtile, axis=(1,2))
    P_pred = np.quantile(pred, q=qtile, axis=(1,2))
    return np.mean(P_pred - P_tar, axis=0)


def latitude_weighting_factor(j, num_lat, s):
    return num_lat * np.cos(3.1416/180. * lat_np(j, num_lat))/s


def weighted_rmse(pred, target):
    num_lat = pred.shape[2]
    lat_t = np.arange(0, num_lat)

    s = np.sum(np.cos(3.1416/180. * lat_np(lat_t, num_lat)))
    weight = np.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = np.sqrt(np.nanmean(weight * (pred - target)**2., axis=(-1,-2)))
    return result


def weighted_acc(pred, target):

    num_lat = np.shape(pred)[2]
    lat_t = np.arange(0, num_lat)
    s = np.sum(np.cos(3.1416/180. * lat_np(lat_t, num_lat)))
    weight = np.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1,1,-1,1))
    result = np.nansum(weight*pred*target, axis=(-1,-2)) / np.sqrt(np.nansum(weight*pred*pred, axis=(-1,-2)) * np.nansum(weight*target*target, axis=(-1,-2)))
    return result



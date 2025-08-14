import numpy as np
import scipy.io
import h5py
from icecream import ic

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds



class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.shape[0]


        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*ops.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return ops.mean(all_norms)
            else:
                return ops.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.shape[0]

        diff_norms = ops.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = ops.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return ops.mean(diff_norms/y_norms)
            else:
                return ops.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class channel_wise_LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, scale=False):
        super(channel_wise_LpLoss, self).__init__()

        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.scale = scale
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[1]

        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*ops.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return ops.mean(all_norms)
            else:
                return ops.sum(all_norms)

        return all_norms

    def rel(self, x, y):

        num_examples = x.shape[0]
        num_channels = x.shape[1]

        x = x.reshape(num_examples, num_channels, -1)
        y = y.reshape(num_examples, num_channels, -1)

        diff_norms = ops.norm(x.reshape(num_examples, num_channels, -1) - y.reshape(num_examples, num_channels, -1), self.p, 2)

        y_norms = ops.norm(y.reshape(num_examples, num_channels, -1), self.p, 2)

        if self.reduction:
            if self.size_average:
                if self.scale:
                    channel_mean = ops.mean(diff_norms/y_norms, 0) 
                    scale_w = channel_mean[0] / channel_mean
                    channel_scale = ops.sum(scale_w * channel_mean)
                    return channel_scale, channel_mean*scale_w 
                else:
                    channel_mean = ops.mean(diff_norms/y_norms, 0)
                    return ops.mean(diff_norms/y_norms), channel_mean
            else:
                if self.scale:
                    channel_sum = ops.sum(diff_norms/y_norms, 0)
                    scale_w = channel_sum[0] / channel_sum
                    channel_sum_scale = ops.sum(scale_w * channel_sum)
                    return channel_sum_scale, channel_sum*scale_w
                else:
                    channel_sum = ops.sum(diff_norms/y_norms, 0)
                    return ops.sum(diff_norms/y_norms), channel_sum
        else:
            return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

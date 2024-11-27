import sys
import os
import torch
from torch import nn
from os import path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))    # add parent dir
from models.model_utils import compute_normalization_parameters

class NonlinBlock(nn.Module):
    def __init__(self, d_in, d_out, batchnorm):
        super(NonlinBlock, self).__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.relu = nn.ReLU()
        self.do_batchnorm = batchnorm
        if batchnorm:
            self.norm_fn = nn.BatchNorm1d(d_out)
        # self.layer_norm = nn.LayerNorm(d_out)

    def forward(self, x):
        x = self.fc(x)
        if self.do_batchnorm:
            x = self.norm_fn(x)
        x = self.relu(x)
        return x


class NonlinMapping(nn.Module):
    def __init__(self, d, layers=2, normalization_params=None, batchnorm=False):
        super(NonlinMapping, self).__init__()
        self.nonlin_mapping = nn.Sequential(*[NonlinBlock(d, d, batchnorm) for i in range(layers - 1)])
        if layers > 0:
            self.lin_mapping = nn.Linear(d, d)
        else:
            self.lin_mapping = lambda x: torch.zeros_like(x)

        self.register_buffer('norm_mean', torch.zeros(d))
        self.register_buffer('norm_sigma', torch.ones(d))

    def initialize_normalization(self, normalization_params):
        if normalization_params is not None:
            if len(normalization_params) > 0:
                self.norm_mean.data.copy_(normalization_params[0])
            if len(normalization_params) > 1:
                self.norm_sigma.data.copy_(normalization_params[1])

    def forward(self, x):
        x = (x - self.norm_mean) / self.norm_sigma
        res = self.nonlin_mapping(x)
        res = self.lin_mapping(res)
        return x + res 


class EmbeddingsMapping(nn.Module):
    def __init__(self, d, video_layers=2, text_layers=2, drop_layers=1, learnable_drop=False, normalization_dataset=None, batchnorm=False):
        super(EmbeddingsMapping, self).__init__()
        self.video_mapping = NonlinMapping(d, video_layers, batchnorm=batchnorm)
        self.text_mapping = NonlinMapping(d, text_layers, batchnorm=batchnorm)

        if learnable_drop:
            self.drop_mapping = NonlinMapping(d, drop_layers, batchnorm=batchnorm)

        if normalization_dataset is not None:
            norm_params = compute_normalization_parameters(normalization_dataset)
            self.video_mapping.initialize_normalization(norm_params[:2])
            self.text_mapping.initialize_normalization(norm_params[2:])

    def map_video(self, x):
        return self.video_mapping(x)

    def map_text(self, z):
        return self.text_mapping(z)

    def compute_distractors(self, v):
        return self.drop_mapping(v)

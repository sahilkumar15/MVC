import os
import os.path as osp

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet

from Modules.diffusion.sampler import KDiffusion, LogNormalDistribution
from Modules.diffusion.modules import Transformer1d, StyleTransformer1d
from Modules.diffusion.diffusion import AudioDiffusionConditional

from Modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator

from mamba_ssm.modules.mamba_simple import Mamba  
from munch import Munch
import yaml

class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif self.layer_type == 'half':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)
            
    def forward(self, x):
        return self.conv(x)

class LearnedUpSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        
        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, output_padding=(1, 0), padding=(1, 0))
        elif self.layer_type == 'half':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, output_padding=1, padding=1)
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance
    
    
class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384, activation=nn.ReLU):
        super().__init__()
        self.dim_in = dim_in
        self.style_dim = style_dim
        
        # Convolution layers for feature extraction
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(1, dim_in, kernel_size=3, stride=1, padding=1)),
            activation(),
            spectral_norm(nn.Conv2d(dim_in, dim_in * 2, kernel_size=3, stride=2, padding=1)),
            activation(),
            spectral_norm(nn.Conv2d(dim_in * 2, max_conv_dim, kernel_size=3, stride=2, padding=1)),
            activation(),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )

        # Learnable style projection
        self.style_proj = nn.Linear(max_conv_dim, style_dim)
        self.bias = nn.Parameter(torch.zeros(style_dim))

    def forward(self, x):
        # Extract features
        features = self.conv(x)  # [B, max_conv_dim, 1, 1]
        features = features.view(features.size(0), -1)  # [B, max_conv_dim]

        # Apply style projection
        style = self.style_proj(features) + self.bias  # [B, style_dim]
        style = F.relu(style)  # Use ReLU as activation

        return style

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=1, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, num_domains, 1, 1, 0))]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        features = []
        for l in self.main:
            x = l(x)
            features.append(x) 
        out = features[-1]
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out, features

    def forward(self, x):
        out, features = self.get_feature(x)
        out = out.squeeze()  # (batch)
        return out, features

class ResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none', dropout_p=0.2):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample_type = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.dropout_p = dropout_p
        
        if self.downsample_type == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.Conv1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1))

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def downsample(self, x):
        if self.downsample_type == 'none':
            return x
        else:
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool1d(x, 2)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv1(x)
        x = self.pool(x)
        if self.normalize:
            x = self.norm2(x)
            
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
    
class BiMambaTextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, style_dim, actv=nn.LeakyReLU(0.2), dropout=0.2, use_checkpointing=True):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        # Optimized convolution layers with grouped and depthwise convolutions
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels // 4)),
                nn.GroupNorm(4, channels),
                actv,
                nn.Dropout(dropout),
            ) for _ in range(depth)
        ])

        # Bi-Mamba blocks (forward and backward) with reduced complexity
        self.mamba_f = Mamba(d_model=channels)
        self.mamba_b = Mamba(d_model=channels)

        # Efficient cross-attention
        self.cross_attention = nn.Linear(2 * channels, 2 * channels, bias=False)

        # Gated mechanism for residual connections
        self.gate = nn.Sequential(
            nn.Conv1d(2 * channels, 2 * channels, kernel_size=1, groups=2),
            nn.SiLU()
        )

        # AdaLayerNorm for style conditioning
        self.adaln = nn.LayerNorm(2 * channels)

        # Projection layer to combine forward and backward outputs
        self.projection = nn.Linear(2 * channels, channels, bias=False)

        self.use_checkpointing = use_checkpointing

    def forward(self, x, style, input_lengths, m):
        # Embedding lookup with efficient padding mask
        x = self.embedding(x).transpose(1, 2)  # [B, emb, T]

        # Apply optimized CNN layers
        for c in self.cnn:
            if self.use_checkpointing and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(c, x)
            else:
                x = c(x)
            x.masked_fill_(m.unsqueeze(1), 0.0)

        # Bi-Mamba processing
        x = x.transpose(1, 2)  # [B, T, chn]
        forward_output = self.mamba_f(x)
        backward_output = self.mamba_b(torch.flip(x, dims=[1]))

        # Combine forward and backward outputs
        combined = torch.cat([forward_output, backward_output], dim=-1)  # [B, T, 2 * chn]

        # Apply Gated Mechanism
        gated_features = self.gate(combined.transpose(-1, -2)).transpose(-1, -2) * combined

        # Cross-Attention
        combined = self.cross_attention(combined)
        combined += gated_features

        # Apply style conditioning
        combined = self.adaln(combined)

        # Project back to original channel dimensions
        combined = self.projection(combined)

        # Mask again to remove padding artifacts
        combined = combined.masked_fill(m.unsqueeze(-1), 0.0)

        return combined

    def inference(self, x):
        # Efficient inference with checkpointing for memory savings
        x = self.embedding(x).transpose(1, 2)
        for c in self.cnn:
            x = c(x)
        x = x.transpose(1, 2)
        forward_output = self.mamba_f(x)
        backward_output = self.mamba_b(torch.flip(x, dims=[1]))
        combined = torch.cat([forward_output, backward_output], dim=-1)
        combined = self.projection(combined)
        return combined

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')

class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))
        
        
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
    
class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
                
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        
        
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class TemporalBiMambaEncoder(nn.Module):
    def __init__(self, channels, style_dim, kernel_size=3, depth=4, dropout=0.2, use_checkpointing=True):
        super().__init__()
        self.channels = channels
        self.style_dim = style_dim

        # Optimized convolutional blocks
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2, groups=channels // 4)),
                nn.GroupNorm(4, channels),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ) for _ in range(depth)
        ])

        # Bi-Mamba blocks for efficient sequence modeling
        self.forward_ssm = Mamba(d_model=channels)
        self.backward_ssm = Mamba(d_model=channels)

        # Optimized cross-attention for temporal consistency
        self.cross_attention = nn.Linear(2 * channels, 2 * channels, bias=False)

        # Residual gating mechanism
        self.gate = nn.Sequential(
            nn.Conv1d(2 * channels, 2 * channels, kernel_size=1, groups=2),
            nn.SiLU()
        )

        # Style modulation with LayerNorm
        self.adaln = nn.LayerNorm(2 * channels)

        # Final projection to match channel dimensions
        self.fusion_proj = nn.Linear(2 * channels, channels, bias=False)

        self.use_checkpointing = use_checkpointing

    def forward(self, x, style, m):
        # Multi-scale hierarchical feature extraction
        for block in self.conv_blocks:
            if self.use_checkpointing and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
            x.masked_fill_(m.unsqueeze(1), 0.0)

        # Bi-Mamba processing
        forward_features = self.forward_ssm(x)
        backward_features = self.backward_ssm(torch.flip(x, dims=[-1]))

        # Reverse backward features
        backward_features = torch.flip(backward_features, dims=[-1])

        # Concatenate forward and backward features
        bidirectional_features = torch.cat([forward_features, backward_features], dim=1)

        # Residual gating
        gated_features = self.gate(bidirectional_features) * bidirectional_features

        # Cross-Attention for temporal context
        bidirectional_features = self.cross_attention(bidirectional_features)
        bidirectional_features += gated_features

        # Style modulation
        fused_features = self.adaln(bidirectional_features)

        # Final fusion projection
        fused_features = self.fusion_proj(fused_features)

        # Apply masking
        fused_features.masked_fill_(m.unsqueeze(1), 0.0)

        return fused_features

    def inference(self, x, style):
        # Multi-scale hierarchical feature extraction
        for block in self.conv_blocks:
            x = block(x)

        # Forward and backward processing
        forward_features = x
        backward_features = x.flip(dims=[-1])

        for f_block, b_block in zip(self.forward_ssm, self.backward_ssm):
            forward_features = f_block(forward_features)
            backward_features = b_block(backward_features)

        # Reverse backward features back to original order
        backward_features = backward_features.flip(dims=[-1])

        # Concatenate forward and backward features
        bidirectional_features = torch.cat([forward_features, backward_features], dim=1)

        # Residual gating
        gated_features = self.gate(bidirectional_features) * bidirectional_features

        # Cross-Attention for refined temporal context
        bidirectional_features = bidirectional_features.transpose(0, 1)  # [T, B, 2 * chn]
        attention_output, _ = self.cross_attention(bidirectional_features, bidirectional_features, bidirectional_features)
        bidirectional_features = gated_features + attention_output.transpose(0, 1)

        # Style modulation
        bidirectional_features = self.adaln(bidirectional_features.transpose(-1, -2), style).transpose(-1, -2)

        # Final fusion projection
        fused_features = self.fusion_proj(bidirectional_features.transpose(-1, -2)).transpose(-1, -2)

        return fused_features

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask



class ExpressiveMambaEncoder(nn.Module):
    def __init__(self, channels, style_dim, kernel_size=3, depth=4, dropout=0.2, use_checkpointing=True):
        super().__init__()
        self.channels = channels
        self.style_dim = style_dim
        self.use_checkpointing = use_checkpointing

        # Optimized Gated Spectrogram Transformation with Depthwise Separable Convolutions
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2, groups=channels // 4)),
                nn.GroupNorm(4, channels),
                nn.SiLU(),
                nn.Dropout(dropout)
            ) for _ in range(depth)
        ])

        # Mamba Blocks for Sequence Modeling
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=channels) for _ in range(depth)
        ])

        # Optimized Residual Gating with Grouped Convolutions
        self.gate = nn.Sequential(
            nn.Conv1d(2 * channels, 2 * channels, kernel_size=1, groups=2, bias=False),
            nn.SiLU()
        )

        # LayerNorm for Lightweight Style Conditioning
        self.adaln = nn.LayerNorm(2 * channels)

        # Final Projection Layer
        self.projection = nn.Linear(2 * channels, channels, bias=False)

    def forward(self, x, style, m):
        # Optimized Gated Spectrogram Transformation
        gated_features = []
        for block in self.gated_convs:
            if self.use_checkpointing and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
            gated_features.append(x)

        # Combine Gated Features
        gated_features = sum(gated_features)

        # Mamba Block Processing
        mamba_features = 0
        for block in self.mamba_blocks:
            mamba_features += block(gated_features)

        # Combine gated and Mamba features
        combined_features = torch.cat([gated_features, mamba_features], dim=1)

        # Residual Gating
        gated_output = self.gate(combined_features) * combined_features

        # Apply Style Modulation
        gated_output = self.adaln(gated_output)

        # Final Projection
        projected_output = self.projection(gated_output)

        # Apply Masking
        projected_output.masked_fill_(m.unsqueeze(1), 0.0)

        return projected_output

    def inference(self, x, style):
        # Efficient Inference
        gated_features = sum(block(x) for block in self.gated_convs)
        mamba_features = sum(block(gated_features) for block in self.mamba_blocks)
        combined_features = torch.cat([gated_features, mamba_features], dim=1)
        gated_output = self.gate(combined_features) * combined_features
        projected_output = self.projection(self.adaln(gated_output))
        return projected_output

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask




    
def load_F0_models(path):
    # load F0 model

    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, map_location='cpu')['net']
    F0_model.load_state_dict(params)
    _ = F0_model.train()
    
    return F0_model

def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    # load ASR model
    def _load_config(path):
        with open(path) as f:
            config = yaml.safe_load(f)
        model_config = config['model_params']
        return model_config

    def _load_model(model_config, model_path):
        model = ASRCNN(**model_config)
        params = torch.load(model_path, map_location='cpu')['model']
        model.load_state_dict(params)
        return model

    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model = _load_model(asr_model_config, ASR_MODEL_PATH)
    _ = asr_model.train()

    return asr_model

def build_model(args, text_aligner, pitch_extractor, bert):
    assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'
    
    if args.decoder.type == "istftnet":
        from Modules.istftnet import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes, 
                gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size) 
    else:
        from Modules.hifigan import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes) 
        
    text_encoder = BiMambaTextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)
    
    
    temporal_encoder = TemporalBiMambaEncoder(
                                        channels=args.hidden_dim,
                                        style_dim=args.style_dim,
                                        kernel_size=3,
                                        depth=args.n_layer,
                                        dropout=args.dropout
                                    )
    
    style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim, activation=nn.ReLU)

    predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) 
        
    # define diffusion model
    if args.multispeaker:
        transformer = StyleTransformer1d(channels=args.style_dim*2, 
                                    context_embedding_features=bert.config.hidden_size,
                                    context_features=args.style_dim*2, 
                                    **args.diffusion.transformer)
    else:
        transformer = Transformer1d(channels=args.style_dim*2, 
                                    context_embedding_features=bert.config.hidden_size,
                                    **args.diffusion.transformer)
    
    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=args.diffusion.embedding_mask_proba, # Conditional dropout of batch elements,
        channels=args.style_dim*2,
        context_features=args.style_dim*2,
    )
    
    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(mean = args.diffusion.dist.mean, std = args.diffusion.dist.std),
        sigma_data=args.diffusion.dist.sigma_data, # a placeholder, will be changed dynamically when start training diffusion model
        dynamic_threshold=0.0 
    )
    diffusion.diffusion.net = transformer
    diffusion.unet = transformer

    
    nets = Munch(
            bert=bert,
            bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),

            predictor=temporal_encoder,
            decoder=decoder,
            text_encoder=text_encoder,

            predictor_encoder=predictor_encoder,
            style_encoder=style_encoder,
            diffusion=diffusion,

            text_aligner = text_aligner,
            pitch_extractor=pitch_extractor,

            mpd = MultiPeriodDiscriminator(),
            msd = MultiResSpecDiscriminator(),
        
            # slm discriminator head
            wd = WavLMDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel),
       )
    
    return nets

def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location='cpu')
    params = state['net']
    for key in model:
        if key in params and key not in ignore_modules:
            print('%s loaded' % key)
            model[key].load_state_dict(params[key], strict=False)
    _ = [model[key].eval() for key in model]
    
    if not load_only_params:
        epoch = state["epoch"]
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
    else:
        epoch = 0
        iters = 0
        
    return model, optimizer, epoch, iters

from __future__ import annotations
import torch
import torch.nn as nn
from libs.model.vqvae.config import VQVAEBaseConfig

class Encoder(nn.Module):
    def __init__(
        self,
        d_val: int,
        d_cond: int,
        config: VQVAEBaseConfig,
    ):
        super().__init__()
        self.config = config
        
        blocks = []
        filter_t, pad_t = config.stride_t * 2, config.stride_t // 2

        # in_ch = input_emb_width + cond_dim
        in_ch = d_val + d_cond
        blocks.append(nn.Conv1d(in_ch, config.d_hidden, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(config.down_t):
            block = nn.Sequential(
                nn.Conv1d(
                    config.d_hidden, config.d_hidden, filter_t, config.stride_t, pad_t),
                Resnet1D(
                    config.d_hidden, 
                    config.depth, 
                    config.dilation_growth_rate, 
                    reverse_dilation=True,
                    activation=config.activation, 
                    norm=config.norm
                ),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(config.d_hidden, config.d_latent, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        if config.is_add_last_ln:
            self.last_ln = nn.LayerNorm(config.d_latent)

    def forward(self, x, cond=None):
        # x: [B, T, Cx]; cond: None or [B, T, Cc]
        if cond is not None:
            x = torch.cat([x, cond], dim=-1)
        x = x.permute(0, 2, 1).contiguous()  # [B, C, T]
        x = self.model(x)
        x = x.permute(0, 2, 1).contiguous()  # [B, T', D]
        if self.config.is_add_last_ln:
            x = self.last_ln(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        d_val: int,
        d_cond: int,
        config: VQVAEBaseConfig
    ):
        super().__init__()
        blocks = []
        
        in_ch = config.d_latent + d_cond
        filter_t, pad_t = config.stride_t * 2, config.stride_t // 2
        blocks.append(nn.Conv1d(in_ch, config.d_hidden, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(config.down_t):
            block = nn.Sequential(
                Resnet1D(
                    config.d_hidden, 
                    config.depth, 
                    config.dilation_growth_rate, 
                    reverse_dilation=True,
                    activation=config.activation, 
                    norm=config.norm
                ),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(config.d_hidden, config.d_hidden, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(config.d_hidden, config.d_hidden, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(config.d_hidden, d_val, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, z, cond=None):
        # z: [B, Tz, Dz]; cond: None or [B, T, Cc]
        if cond is not None:
            z = torch.cat([z, cond], dim=-1)
        z = z.permute(0, 2, 1).contiguous()  # [B, C, Tz]
        y = self.model(z)
        y = y.permute(0, 2, 1).contiguous()  # [B, T, pose_dim]
        return y

class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)

class ResConv1DBlock(nn.Module):
    def __init__(
        self,
        n_in, 
        n_state, 
        dilation=1, 
        activation='silu', 
        norm=None, 
        dropout=None
    ):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
            
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
            
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
            
        

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)     


    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)
            
        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x

class Resnet1D(nn.Module):
    def __init__(
        self, 
        n_in, 
        n_depth, 
        dilation_growth_rate=1, 
        reverse_dilation=True, 
        activation='relu', 
        norm=None
    ):
        super().__init__()
        
        blocks = [
            ResConv1DBlock(
                n_in, 
                n_in, 
                dilation=dilation_growth_rate ** depth, 
                activation=activation, norm=norm) 
            for depth in range(n_depth)
        ]

        if reverse_dilation:
            blocks = blocks[::-1]
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x):        
        return self.model(x)

from Models.TAU.modules.modules import (ConvSC, TAUSubBlock)

import src.Metric as Metric
# import src.ASTD_DataLoader as ASTD_DataLoader
import src.Preprocessing as Preprocessing
from Data.UrbanFM import get_dataloader, get_data, check_visualization

import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import torch.nn.functional as F
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import os

# from einops.layers.torch import Rearrange

# 모든 경고 메시지를 무시
warnings.filterwarnings("ignore")

#%%

class SimVP_Model(nn.Module):

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=8, N_T=2, model_type='tau',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
            input_resolution=(H, W), model_type=model_type,
            mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        
        return Y

#%%

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

#%%

class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y

#%%

class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type='tau',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = TAUSubBlock(
            in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
            drop=drop, drop_path=drop_path, act_layer=nn.GELU)

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)

class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type='tau',
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y

#%%

def diff_div_reg(pred_y, batch_y, tau=0.1, eps=1e-12):
    B, T, C = pred_y.shape[:3]
    if T <= 2:  return 0
    gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
    gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
    softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
    softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
    loss_gap = softmax_gap_p * \
        torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
    return loss_gap.mean()

#%%
###################################### Set Device ######################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

#%%

###################################### Spatial-Temporal Data Load ######################################

dataname = 'P2'
datapath = f'Data/UrbanFM/{dataname}/'

dt = get_data(datapath)
_, width, height = dt.shape
n_features = 1

check_visualization(dt[0, :, :])

seq_input = 30
seq_output = 720

X, Y = Preprocessing.seq_data(dt, seq_input, seq_output)
X = X.permute(0, 1, 4, 2, 3).contiguous()
Y = Y.permute(0, 1, 4, 2, 3).contiguous()
print(X.size(), Y.size())

###################################### Data Split ######################################

split = int(X.size(0) * 0.8)

x_train = X[:split]    
y_train = Y[:split]    

x_test = X[split:].to(device)
y_test = Y[split:].to(device)

print(x_train.size(), y_train.size())
print(x_test.size(), y_test.size())

del X
del Y

###################################### Model Option ######################################

in_shape = [seq_input, n_features, width, height]
alpha = 0.1

model = SimVP_Model(in_shape)
model.to(device)

###################################### Learning Option ######################################

batch_size = 16
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

###################################### Train ######################################

Epoch = 1000

best_mse = 10**5
best_mae = 10**5
best_psnr = -1 * 10**5
best_ssim = -1 * 10**5

for epoch in range(Epoch):
    
    model.train()
    
    for i in range(0, x_train.size(0), batch_size):
        
        if i + batch_size > x_train.size(0):
            x, y = x_train[i:].to(device), y_train[i:].to(device)
        else:
            x, y = x_train[i:(i+batch_size)].to(device), y_train[i:(i+batch_size)].to(device)

        optimizer.zero_grad()
        output = model(x)
        
        loss = F.mse_loss(output, y[:, -1*seq_input:]) + alpha + diff_div_reg(output, y)
        loss.backward()
        
        optimizer.step()
    
    outputs, actuals = [], []
    
    model.eval()
    with torch.no_grad():
        
        for i in range(0, x_test.size(0), batch_size):
            
            if i + batch_size > x_test.size(0):
                x, y = x_test[i:].to(device), y_test[i:].to(device)
            else:
                x, y = x_test[i:(i+batch_size)].to(device), y_test[i:(i+batch_size)].to(device)
                
            output = model(x)
            
            outputs.append(output)
            actuals.append(y)
        
        outputs = torch.cat(outputs, dim=0).permute(0, 1, 3, 4, 2).contiguous()
        actuals = torch.cat(actuals, dim=0).permute(0, 1, 3, 4, 2).contiguous()
        
        mse, mae, psnr, ssim = Metric.metric(outputs, actuals, n_features)
        
        # a = actuals[-1, -1, :, :, 0].cpu()
        # p = outputs[-1, -1, :, :, 0].cpu()
        p = torch.sum(outputs, dim=(2, 3, 4)).cpu()
        a = torch.sum(actuals, dim=(2, 3, 4)).cpu()
    
        if best_mse > mse: 
            best_mse = mse
            torch.save(model.state_dict(), f'Models/TAU/models/TAU_{seq_output}_{dataname}_mse')
        if best_mae > mae: 
            best_mae = mae
            torch.save(model.state_dict(), f'Models/TAU/models/TAU_{seq_output}_{dataname}_mae')
        if best_psnr < psnr: 
            best_psnr = psnr
            torch.save(model.state_dict(), f'Models/TAU/models/TAU_{seq_output}_{dataname}_psnr')
        if best_ssim < ssim: 
            best_ssim= ssim
            torch.save(model.state_dict(), f'Models/TAU/models/TAU_{seq_output}_{dataname}_ssim')
        
        if epoch%10 == 0:
                plt.plot(p[0], label='prediction', color='r')
                plt.plot(a[0], label='actual', color='b')
                plt.legend()
                plt.show()
                
        print(f'[Model: TAU / Term: {seq_output} / Epoch: {epoch}]')
        print(f'[B_MSE: {best_mse:.9f} / B_MAE: {best_mae:.8f} / B_PSNR: {best_psnr:.4f} / B_SSIM: {best_ssim:.4f}]')
        print(f'[MSE: {mse:.9f} / MAE: {mae:.8f} / PSNR: {psnr:.4f} / SSIM: {ssim:.4f}]')
        
#%%

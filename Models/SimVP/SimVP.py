
from Models.SimVP.modules import ConvSC, Inception

import src.Metric as Metric
import src.Preprocessing as Preprocessing
from Data.MMNIST import MovingMNIST, Test_visualization

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

from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import argparse

# 모든 경고 메시지를 무시
warnings.filterwarnings("ignore")

#%%

def stride_generator(N, reverse=False):
    
    strides = [1, 2]*10
    
    if reverse: return list(reversed(strides[:N]))  # decoding시 stride 반대로
    else: return strides[:N]

#%%

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):    
        # N_S - Convolution encoding 횟수 / C_in - channel크기 / C_hid - hidden dimension
        super(Encoder,self).__init__()
        
        strides = stride_generator(N_S)
        
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):
        # B*T, C, W, H
    
        enc1 = self.enc[0](x)
        #print(f'enc1: {enc1.size()}')
        latent = enc1
        
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            #print(f'{i}th ConvSC: {latent.size()}')
            
        return latent, enc1

#%%

class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        
        strides = stride_generator(N_S, reverse=True)       # reverse: 차원되돌리기(ConvTranspose)
        
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
            #print(f'{i}th dec ConvSC: {hid.size()}')
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1)) # skip connection
        #print(f'last ConvSC: {Y.size()}')
        Y = self.readout(Y)
        #print(f'Decoder Output: {Y.size()}')
        
        return Y

#%%

class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T      # 얼마나 깊게 group convolution encoding & group convolution decoding 할건지
        
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)   # N_T번 x incep_ker 개수 만큼 group convolution
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        #print(f'Mid_Xnet input: {x.size()}')
        x = x.reshape(B, T*C, H, W)
        #print(f'x reshape: {x.size()}')
        
        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            #print(f'Mid encoder z size: {z.size()}')
            if i < self.N_T - 1:
                #print('in')
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            #print(f'Mid decoder z size: {torch.cat([z, skips[-i]], dim=1).size()}')
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))   # skip connection

        y = z.reshape(B, T, C, H, W)
        return y

#%%

class SimVP(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(SimVP, self).__init__()
        
        T, C, H, W = shape_in
        
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        #print(f'Model Input: {x_raw.size()}')
        x = x_raw.view(B*T, C, H, W)
        #print(f'Encoder Input: {x.size()}')

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        
        #print(f'After Encoder: {embed.size()}')
        z = embed.view(B, T, C_, H_, W_)
        #print(f'Mid Input: {z.size()}')
        hid = self.hid(z)
        #print(f'Mid Output: {hid.size()}')
        hid = hid.reshape(B*T, C_, H_, W_)
        
        #print(f'Decoder Skip size: {skip.size()}')
        #print(f'Decoder Input: {hid.size()}')
        Y = self.dec(hid, skip)
        #print(f'Decoder Output: {Y.size()}')
        Y = Y.reshape(B, T, C, H, W)
        #print(f'Final: {Y.size()}')
        
        return Y
    
#%%

###################################### Set Device ######################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

#%%

###################################### Spatial-Temporal Data Load ######################################

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='Data/MMNIST/')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
args = parser.parse_args()

seq_input = 10
seq_output = 10

mm = MovingMNIST(root=args.root, is_train=True, n_frames_input=seq_input, n_frames_output=seq_output, num_objects=[2])
train_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=args.batch_size, shuffle=True, num_workers=0)

mm = MovingMNIST(root=args.root, is_train=False, n_frames_input=seq_input, n_frames_output=seq_output, num_objects=[2])
test_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=args.batch_size, shuffle=False, num_workers=0)

n_features = 1

Test_visualization(train_loader, test_loader, args.batch_size, seq_input, seq_output)

###################################### Model Option ######################################

width, height = 64

in_shape = [seq_input, n_features, width, height]
hid_S = 16
hid_T = int(seq_input * hid_S * (seq_output / seq_input)) 

model = SimVP(in_shape)#, hid_S=hid_S, hid_T=hid_T)
model.to(device)

###################################### Learning Option ######################################

batch_size = args.batch_size
lr = 0.00015
optimizer = optim.Adam(model.parameters(), lr=lr)

###################################### Train ######################################

Epoch = 1000

best_mse = 10**5
best_mae = 10**5
best_psnr = -1 * 10**5
best_ssim = -1 * 10**5

mses = []

for epoch in range(Epoch):
    
    model.train()
    
    for data in train_loader:
        
        optimizer.zero_grad()
        output = model(data[1].cuda())
        
        loss = F.mse_loss(output, data[2].cuda())
        loss.backward()
        
        optimizer.step()
    
    mses, maes, ssims, psnrs = [], [], [], []
    
    model.eval()
    with torch.no_grad():
        
        for data in test_loader:
            
            output = model(data[1].cuda())
            output = output.permute(0, 1, 3, 4, 2).contiguous()
            actual = data[2].permute(0, 1, 3, 4, 2).contiguous()
            
            mse, mae, ssim, psnr = Metric.compute_metrics(output, actual)
            
            mses.append(mse)
            maes.append(mae)
            ssims.append(ssim)
            psnrs.append(psnr)
        
        mse = np.mean(mses)
        mae = np.mean(maes)
        ssim = np.mean(ssims)
        psnr = np.mean(psnrs)
        
        a = actual[-1, -1, :, :, 0].cpu()
        p = output[-1, -1, :, :, 0].cpu()
    
        if best_mse > mse: best_mse = mse
        if best_mae > mae: best_mae = mae
        if best_psnr < psnr: best_psnr = psnr
        if best_ssim < ssim: best_ssim= ssim
        
        if epoch%10 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            im1 = ax1.imshow(a, cmap='viridis', aspect='auto')
            fig.colorbar(im1, ax=ax1)  
            im2 = ax2.imshow(p, cmap='viridis', aspect='auto')
            fig.colorbar(im2, ax=ax2)
            
            plt.show()
                
        print(f'[Model: SimVP / Term: {seq_output} / Epoch: {epoch}]')
        print(f'[B_MSE: {best_mse:.9f} / B_MAE: {best_mae:.8f} / B_PSNR: {best_psnr:.4f} / B_SSIM: {best_ssim:.4f}]')
        print(f'[MSE: {mse:.9f} / MAE: {mae:.8f} / PSNR: {psnr:.4f} / SSIM: {ssim:.4f}]')
        
        # mses.append(mse)
        #np.save(modelsave_path + f'/mses_{seq_output[Term]}_{n_try}.npy', np.array(mses))
        
#%%



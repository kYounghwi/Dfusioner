
import src.Metric as Metric
import src.Preprocessing as Preprocessing
from Data.UrbanFM import get_dataloader, get_data, check_visualization, check_visualization2

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

def caculate_map(width, kernel, padding, stride):
    return ((width-kernel+2*padding)/stride)+1

#%%

class _CDDNet(nn.Module):

    def __init__(self, width, height, window_size, initial_embed, n_features,
                 encode_embed, encode_kernel_size, encode_stride, encode__padding, 
                 kernel_size, padding, n_layers=3, 
                 dropout=0.0, encode_dropout=0.0, aggregation_method='concat'):
        super(_CDDNet, self).__init__()
        
        self.width = int(caculate_map(caculate_map(width,encode_kernel_size[0],encode__padding[0],encode_stride[0]),encode_kernel_size[1],encode__padding[1],encode_stride[1]))
        self.height = self.width
        
        self.embed = nn.Linear(n_features, initial_embed)
        
        self.encoder = Encoder(initial_embed, encode_embed, encode_kernel_size, encode_stride, encode__padding, encode_dropout)
        self.decoder = Decoder(n_features, encode_embed, encode_kernel_size, encode_stride, encode__padding, encode_dropout)
        
        self.block = nn.Sequential()
        for i in range(n_layers):
            self.block.add_module(f'block_{i}', Block(window_size, self.width, self.height, encode_embed[1], kernel_size, padding, dropout))
        
        self.aggregation = feature_aggregation(encode_embed[1], aggregation_method)
          
    def forward(self, x):   # batch, seq, width, height, class
        
        #seq_last = torch.mean(x, dim=1, keepdim=True).detach()
        
        x = self.embed(x)   # batch, seq, width, height, initial_embed
        
        x = self.encoder(x)     # batch, sequence, W, H, embed_dim
        B, T, W, H, E = x.size()
        x = x.view(B, T, -1, E)     # batch, sequence, W*H, embed_dim
        
        inp = [x, x]
        
        outs = self.block(inp)
        out = self.aggregation(outs)
        out = out.view(B, T, W, H, E)
        
        out = self.decoder(out)
        
        #out = out + seq_last
        
        return out

#%%

class Encoder(nn.Module):

    def __init__(self, initial_embed, embed_dim, kernel_size, stride, padding, dropout):
        super(Encoder, self).__init__()

        self.norm = nn.LayerNorm(initial_embed)
        self.proj = nn.Sequential(
            nn.Conv2d(initial_embed, embed_dim[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0]),
            nn.GELU(),
            nn.Conv2d(embed_dim[0], embed_dim[1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
            )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):   # batch, sequence, width, height, embed_dim
    
        T = x.size(1)
        
        outs = None
        for i in range(T):
        
            out = self.norm(x[:, i]).permute(0, 3, 1, 2)           # batch, embed_dim, W, H
            out = self.proj(out)                    # batch, embed_dim, W, H
            out = self.drop(out.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)  # batch, W, H, embed_dim
        
            if i == 0:
                outs = out
            else:
                outs = torch.cat((outs, out), dim=1)
                
        return outs  # batch, sequence, W, H, embed_dim

#%%

class Decoder(nn.Module):

    def __init__(self, n_features, embed_dim, kernel_size, stride, padding, dropout):
        super(Decoder, self).__init__()

        self.norm = nn.LayerNorm(embed_dim[1])
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(embed_dim[1], embed_dim[0], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1]),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim[0], n_features, kernel_size=kernel_size[0]+1, stride=stride[0], padding=padding[0])
            )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):   # batch, sequence, width, height, embed_dim
        
        T = x.size(1)
        
        outs = None
        for i in range(T):
        
            out = self.norm(x[:, i]).permute(0, 3, 1, 2)           # batch, embed_dim, W, H
            out = self.proj(out)                    # batch, F, W, H
            out = self.drop(out.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)  # batch, W, H, F
        
            if i == 0:
                outs = out
            else:
                outs = torch.cat((outs, out), dim=1)
                
        return outs  # batch, sequence, W, H, F

#%%

class feature_aggregation(nn.Module):
    def __init__(self, embed_dim, aggregation_method):
        super(feature_aggregation, self).__init__()
        
        self.method = aggregation_method
        
        if aggregation_method == 'concat':
            self.projection = nn.Linear(embed_dim*2, embed_dim, bias=False)
        if aggregation_method == 'average':
            self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, outs):
        
        if self.method == 'concat':
            out = torch.cat(outs, dim=-1)
            out = self.projection(out)
            
        if self.method == 'average':
            out = torch.mean(torch.stack(outs, dim=-1), dim=-1)
            out = self.projection(out)
        
        return out

#%%

class Block(nn.Module):
    def __init__(self, window_size, width, height, embed_dim, kernel_size, padding, dropout):
        super(Block, self).__init__()
        
        self.spatial_size = width*height
            
        self.blocks = nn.ModuleList([
            Time_Space(window_size, self.spatial_size, embed_dim),
            Space_Time(window_size, self.spatial_size, embed_dim)
            ])
        
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # [batch, seq, patch, spatial] x 2
        
        outs = []
        for i in range(2):
            outs.append(self.drop(self.blocks[i](x[i])))
            
        return outs

#%%

class Local_Dependency_layer(nn.Module):
    def __init__(self, projection_size, embed_dim):
        super(Local_Dependency_layer, self).__init__()
        
        self.embed_channels = embed_dim
        
        # embedding-wise temporal layer
        self.layer = nn.ModuleList()       
        for i in range(embed_dim):
            self.layer.append(nn.Linear(projection_size, projection_size))

    def forward(self, x):   # batch, ?, projection_size, embed
        
        outs = []
        
        for i in range(self.embed_channels):
            outs.append(self.layer[i](x[:, :, :, i]))
            
        out = torch.stack(outs, dim=-1)
       
        return out

#%%

class Time_Space(nn.Module):
    def __init__(self, window_size, spatial_size, embed_dim):
        super(Time_Space, self).__init__()
        
        self.localtime = Local_Dependency_layer(window_size, embed_dim)
        self.localspace = Local_Dependency_layer(spatial_size, embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):   # batch, sequence, spatial, embed

        out = self.localtime(x.permute(0, 2, 1, 3))   # batch, spatial, sequence, embed
        out = out.permute(0, 2, 1, 3).contiguous()   # batch, sequence, spatial, embed
        
        out = self.localspace(out)   # batch, sequence, spatial, embed

        out = self.norm(out)
        
        return out
    
#%%

class Space_Time(nn.Module):
    def __init__(self, window_size, spatial_size, embed_dim):
        super(Space_Time, self).__init__()
        
        self.localspace = Local_Dependency_layer(spatial_size, embed_dim)
        self.localtime = Local_Dependency_layer(window_size, embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):   # batch, sequence, spatial, embed

        out = self.localspace(x)   # batch, sequence, spatial, embed
        out = out.permute(0, 2, 1, 3).contiguous()   # batch, spatial, sequence, embed
        
        out = self.localtime(out)   # batch, spatial, sequence, embed
        out = out.permute(0, 2, 1, 3).contiguous()   # batch, sequence, spatial, embed

        out = self.norm(out)
        
        return out
    
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
seq_output = 360

X, Y = Preprocessing.seq_data(dt, seq_input, seq_output)
print(X.size(), Y.size())

###################################### Data Split ######################################

split = int(X.size(0) * 0.8)

x_train = X[:split]    
y_train = Y[:split]    

x_test = X[split:].to(device)
y_test = Y[split:].to(device)

print(x_train.size(), y_train.size())
print(x_test.size(), y_test.size())

check_visualization2(x_test)
check_visualization2(y_test)

del X
del Y

###################################### Model Option ######################################

initial_embed = 18

encode_embed = [18, 32]
encode_kernel = [3, 5]
encode_stride = [2, 1]
encode_padding = [1, 2]

kernel_size = [7, 5]
padding = [3, 2]

n_layers = 4
dropout = 0.05
encode_dropout=0.05

aggregation_method = 'concat'

model = _CDDNet(width, height, seq_input, initial_embed, n_features,
                encode_embed, encode_kernel, encode_stride, encode_padding, 
                kernel_size, padding, n_layers, 
                dropout, encode_dropout, aggregation_method)
model.to(device)

###################################### Learning Option ######################################

batch_size = 16
lr = 0.0001
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
    
    for i in range(0, x_train.size(0), batch_size):
        
        if i + batch_size > x_train.size(0):
            x, y = x_train[i:].to(device), y_train[i:].to(device)
        else:
            x, y = x_train[i:(i+batch_size)].to(device), y_train[i:(i+batch_size)].to(device)
        
        optimizer.zero_grad()
        output = model(x)
        
        loss = F.mse_loss(output, y)
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
        
        outputs = torch.cat(outputs, dim=0)
        actuals = torch.cat(actuals, dim=0)
    
        mse, mae, psnr, ssim = Metric.metric(outputs, actuals, n_features)
        
        p = torch.sum(outputs, dim=(2, 3, 4)).cpu()
        a = torch.sum(actuals, dim=(2, 3, 4)).cpu()
    
        if best_mse > mse: 
            best_mse = mse
            torch.save(model.state_dict(), f'Models/DFusioner/models/DFusioner_{seq_output}_{dataname}_mse4')
        if best_mae > mae: best_mae = mae
        if best_psnr < psnr: best_psnr = psnr
        if best_ssim < ssim: best_ssim= ssim
                
        print(f'[Model: DFusioner / Term: {seq_output} / Epoch: {epoch}]')
        print(f'[B_MSE: {best_mse:.9f} / B_MAE: {best_mae:.8f} / B_PSNR: {best_psnr:.4f} / B_SSIM: {best_ssim:.4f}]')
        print(f'[MSE: {mse:.9f} / MAE: {mae:.8f} / PSNR: {psnr:.4f} / SSIM: {ssim:.4f}]')
        
        
#%%



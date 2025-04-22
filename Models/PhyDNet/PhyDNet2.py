
import src.Metric as Metric
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

from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torchvision
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Models.PhyDNet.models.models import ConvLSTM,PhyCell, EncoderRNN
from Models.PhyDNet.constrain_moments import K2M

import argparse

# 모든 경고 메시지를 무시
warnings.filterwarnings("ignore")

#%%    

###################################### Set Device ######################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

#%%

constraints = torch.zeros((49,7,7)).to(device)
ind = 0
for i in range(0,7):
    for j in range(0,7):
        constraints[ind,i,j] = 1
        ind +=1    

def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion, teacher_forcing_ratio): 
               
    encoder_optimizer.zero_grad()
    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
    input_length  = input_tensor.size(1)
    target_length = target_tensor.size(1)
    
    loss = 0
    
    for ei in range(input_length - 1): 
        
        encoder_output, encoder_hidden, output_image, _, _ = encoder(input_tensor[:, ei, :, :, :], (ei==0) )
        loss += criterion(output_image, input_tensor[:, ei+1, :, :, :])

    decoder_input = input_tensor[:, -1, :, :, :] # first decoder input = last image of input sequence
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
    
    for di in range(target_length):
        
        decoder_output, decoder_hidden, output_image, _, _ = encoder(decoder_input)
        target = target_tensor[:, di, :, :, :]
        loss += criterion(output_image, target)
        
        if use_teacher_forcing:
            decoder_input = target # Teacher forcing    
        else:
            decoder_input = output_image

    # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
    k2m = K2M([7,7]).to(device)
    for b in range(0, encoder.phycell.cell_list[0].input_dim):
        filters = encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)     
        m = k2m(filters.double()) 
        m  = m.float()   
        loss += criterion(m, constraints) # constrains is a precomputed matrix   
        
    loss.backward()
    encoder_optimizer.step()
    
    return loss.item() / target_length

#%%

###################################### Spatial-Temporal Data Load ######################################

dataname = 'P1'
datapath = f'Data/UrbanFM/{dataname}/'

dt = get_data(datapath)
_, width, height = dt.shape
n_features = 1

check_visualization(dt[0, :, :])

seq_input = 30
seq_output = 30

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

###################################### Model Option ######################################

phycell  =  PhyCell(input_shape=(8, 8), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
convcell =  ConvLSTM(input_shape=(8, 8), input_dim=64, hidden_dims=[128, 128, 64], n_layers=3, kernel_size=(3,3), device=device)   
encoder  = EncoderRNN(phycell, convcell, device)

###################################### Learning Option ######################################

batch_size = 16
lr = 1e-3

encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=2,factor=0.1,verbose=True)
criterion = nn.MSELoss()

###################################### Train ######################################

Epoch = 1000

best_mse = 10**5
best_mae = 10**5
best_psnr = -1 * 10**5
best_ssim = -1 * 10**5

mses = []

for epoch in range(Epoch):
    
    teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003)
    
    for i in range(0, x_train.size(0), batch_size):
        
        if i + batch_size > x_train.size(0):
            x, y = x_train[i:].to(device), y_train[i:].to(device)
        else:
            x, y = x_train[i:(i+batch_size)].to(device), y_train[i:(i+batch_size)].to(device)
        
        train_on_batch(x, y, encoder, encoder_optimizer, criterion, teacher_forcing_ratio)
    
    outputs, actuals = [], []
    
    encoder.eval()
    with torch.no_grad():
        
        for i in range(0, x_test.size(0), batch_size):
            
            if i + batch_size > x_test.size(0):
                x, y = x_test[i:].to(device), y_test[i:].to(device)
            else:
                x, y = x_test[i:(i+batch_size)].to(device), y_test[i:(i+batch_size)].to(device)
                
            decoder_input = x[:,-1,:,:,:] # first decoder input= last image of input sequence
            predictions = []

            for di in range(seq_input):
                decoder_output, decoder_hidden, output_image, _, _ = encoder(decoder_input, False, False)
                decoder_input = output_image
                predictions.append(output_image.cpu())

            predictions = torch.stack(predictions, dim=0) # (seq, batch_size, 7, 32, 32)
            predictions = predictions.permute(1, 0, 2, 3, 4).contiguous() # (batch_size, seq, 7, 32, 32)
            
            outputs.append(predictions)
            actuals.append(y)
        
        outputs = torch.cat(outputs, dim=0).permute(0, 1, 3, 4, 2).contiguous()
        actuals = torch.cat(actuals, dim=0).permute(0, 1, 3, 4, 2).contiguous()
    
        mse, mae, psnr, ssim = Metric.metric(outputs, actuals, n_features)
        
        a = actuals[-1, -1, :, :, 0].cpu()
        p = outputs[-1, -1, :, :, 0].cpu()
    
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
                
        print(f'[Model: PhyDNet / Term: {seq_output} / Epoch: {epoch}]')
        print(f'[B_MSE: {best_mse:.9f} / B_MAE: {best_mae:.8f} / B_PSNR: {best_psnr:.4f} / B_SSIM: {best_ssim:.4f}]')
        print(f'[MSE: {mse:.9f} / MAE: {mae:.8f} / PSNR: {psnr:.4f} / SSIM: {ssim:.4f}]')
        
        
#%%



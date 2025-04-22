
import src.Metric as Metric
import src.Preprocessing as Preprocessing
from Data.UrbanFM import get_dataloader, get_data, check_visualization

import Models.MAU.core.models.MAU as MAU

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
from numpy import sqrt
import torch.optim.lr_scheduler as lr_scheduler

import argparse

# 모든 경고 메시지를 무시
warnings.filterwarnings("ignore")
    
#%%    

###################################### Set Device ######################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

#%%

def schedule_sampling(eta, itr, channel, batch_size):
    
    zeros = np.zeros((batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    #print('eta: ', eta)
    random_flip = np.random.random_sample(
        (batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_height // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * channel))
    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_height // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * channel))
    return eta, real_input_flag

#%%

###################################### Spatial-Temporal Data Load ######################################

dataname = 'P1'
datapath = f'Data/UrbanFM/{dataname}/'

dt = get_data(datapath)
_, width, height = dt.shape
n_features = 1

check_visualization(dt[0, :, :])

seq_input = 30
seq_output = 360

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

parser = MAU.configs(seq_input, seq_output, width, height, n_features)
args = parser.parse_args()

num_hidden = []
for _ in range(args.num_layers):
    num_hidden.append(args.num_hidden)

model = MAU.RNN(args.num_layers, num_hidden, args)
model.to(device)

###################################### Learning Option ######################################

batch_size = 16
lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

begin = 0
eta = args.sampling_start_value
eta -= (begin * args.sampling_changing_rate)
itr = begin
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
        
        batch_size_ = x.size(0)
        
        eta, real_input_flag = schedule_sampling(eta, itr, args.img_channel, batch_size_)
        real_input_flag = torch.FloatTensor(real_input_flag).to(device)
        inp = torch.cat((x, y), dim=1)
        
        optimizer.zero_grad()
        output = model(inp, real_input_flag)
        
        loss = F.mse_loss(output[:, -seq_input:], y)
        loss.backward()
        
        optimizer.step()
        
        if itr >= args.sampling_stop_iter and itr % args.delay_interval == 0:
            scheduler.step()
        # self.scheduler_F.step()
        # self.scheduler_D.step()
            #print('Lr decay to:%.8f', optimizer.param_groups[0]['lr'])
        
        itr += 1
    
    outputs, actuals = [], []
    
    model.eval()
    with torch.no_grad():
        
        for i in range(0, x_test.size(0), batch_size):
            
            if i + batch_size > x_test.size(0):
                x, y = x_test[i:].to(device), y_test[i:].to(device)
            else:
                x, y = x_test[i:(i+batch_size)].to(device), y_test[i:(i+batch_size)].to(device)
            
            batch_size_ = x.size(0)
            
            test_input_flag = np.zeros((batch_size_, args.total_length - args.input_length - 1,
                     args.img_height // args.patch_size,
                     args.img_width // args.patch_size,
                     args.patch_size ** 2 * args.img_channel))
            test_input_flag = torch.FloatTensor(test_input_flag).to(device)
            inp = torch.cat((x, y), dim=1)
            
            output = model(inp, test_input_flag)
            
            outputs.append(output[:, -seq_input:])
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
            torch.save(model.state_dict(), f'Models/MAU/models/MAU_{seq_output}_{dataname}_mse')
        if best_mae > mae: 
            best_mae = mae
            torch.save(model.state_dict(), f'Models/MAU/models/MAU_{seq_output}_{dataname}_mae')
        if best_psnr < psnr: 
            best_psnr = psnr
            torch.save(model.state_dict(), f'Models/MAU/models/MAU_{seq_output}_{dataname}_psnr')
        if best_ssim < ssim: 
            best_ssim= ssim
            torch.save(model.state_dict(), f'Models/MAU/models/MAU_{seq_output}_{dataname}_ssim')
        
        if epoch%10 == 0:
                plt.plot(p[0], label='prediction', color='r')
                plt.plot(a[0], label='actual', color='b')
                plt.legend()
                plt.show()
                
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # im1 = ax1.imshow(a, cmap='viridis', aspect='auto')
            # fig.colorbar(im1, ax=ax1)  
            # im2 = ax2.imshow(p, cmap='viridis', aspect='auto')
            # fig.colorbar(im2, ax=ax2)
            
            # plt.show()
                
        print(f'[Model: MAU / Term: {seq_output} / Epoch: {epoch}]')
        print(f'[B_MSE: {best_mse:.9f} / B_MAE: {best_mae:.8f} / B_PSNR: {best_psnr:.4f} / B_SSIM: {best_ssim:.4f}]')
        print(f'[MSE: {mse:.9f} / MAE: {mae:.8f} / PSNR: {psnr:.4f} / SSIM: {ssim:.4f}]')
        
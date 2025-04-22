
import numpy as np
from datetime import datetime, timedelta
import torch
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

#%%

def get_dataloader(datapath, scaler_X, scaler_Y, batch_size, mode='train'):
    
    datapath = os.path.join(datapath, mode)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    X = Tensor(np.expand_dims(np.load(os.path.join(datapath, 'X.npy')), 1)) / scaler_X
    Y = Tensor(np.expand_dims(np.load(os.path.join(datapath, 'Y.npy')), 1)) / scaler_Y
    ext = Tensor(np.load(os.path.join(datapath, 'ext.npy')))
    assert len(X) == len(Y)
    print('# {} samples: {}'.format(mode, len(X)))

    data = torch.utils.data.TensorDataset(X, ext, Y)
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader

#%%

def get_data(datapath):
    
    data1 = np.load(datapath+'train'+'/X.npy')
    print(f'train: {data1.shape}')

    data2 = np.load(datapath+'test'+'/X.npy')
    print(f'valid: {data2.shape}')

    data3 = np.load(datapath+'valid'+'/X.npy')
    print(f'test: {data3.shape}')

    data = np.concatenate((data1, data2, data3), axis=0)
    print(f'total: {data.shape}')
    
    d_max = np.max(data)
    d_min = np.min(data)
    
    data = (data-d_min) / (d_max - d_min)
    
    #data = torch.tensor(data).unsqueeze(-1)
    
    return data

#%%

def check_visualization(dt):
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    
    im1 = ax1.imshow(dt, cmap='viridis', aspect='auto')
    fig.colorbar(im1, ax=ax1)
    plt.show()

#%%

def check_visualization2(df):
    
    for i in range(30, 20, -1):
    
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        
        im1 = ax1.imshow(df[0, i-1].cpu(), cmap='viridis', aspect='auto')
        fig.colorbar(im1, ax=ax1)
        plt.show()


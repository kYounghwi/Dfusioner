
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

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

#%%

class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, seq_input, seq_output, input_dim, hidden_dim, kernel_size, num_layers,
                 bias=True):
        super(ConvLSTM, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(hidden_dim[-1], self.input_dim,
                                    kernel_size=1, stride=1, padding=0, bias=False)
        self.linear_last = nn.Linear(seq_input, seq_output)

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        layer_output, last_state_list
        """

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            output_result = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
                output_result.append(self.conv_last(h))
            
            layer_output = torch.stack(output_inner, dim=1)     # list to tensor b, s, c, w, h
            cur_layer_input = layer_output
            
            output_result = torch.stack(output_result, dim=1)   # for match output channel

            #layer_output_list.append(layer_output)
            layer_output_list.append(output_result)
            last_state_list.append([h, c])

        # 마지막 layer만 추출
        #layer_output_list = layer_output_list[-1:]
        layer_output = layer_output_list[-1]
        last_state_list = last_state_list[-1:]

        return layer_output, last_state_list
    
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

hidden_dim = [n_features*4, n_features*4, n_features*4]
kernel_size = (3, 3)
num_layers = 3

model = ConvLSTM(seq_input, seq_output, n_features, hidden_dim, kernel_size, num_layers, bias=True)
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
                
        print(f'[Model: ConvLSTM / Term: {seq_output} / Epoch: {epoch}]')
        print(f'[B_MSE: {best_mse:.9f} / B_MAE: {best_mae:.8f} / B_PSNR: {best_psnr:.4f} / B_SSIM: {best_ssim:.4f}]')
        print(f'[MSE: {mse:.9f} / MAE: {mae:.8f} / PSNR: {psnr:.4f} / SSIM: {ssim:.4f}]')
        
        # mses.append(mse)
        #np.save(modelsave_path + f'/mses_{seq_output[Term]}_{n_try}.npy', np.array(mses))
        
#%%



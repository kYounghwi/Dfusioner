
import numpy as np
from datetime import datetime, timedelta
import torch

###################################### sequence split ######################################

def seq_data(x, seq_len, Term):
  
    x_seq = []
    y_seq = []
    
    for i in range(seq_len, len(x) - Term):
        x_seq.append(x[i-seq_len : i])     
        y_seq.append(x[i+Term-seq_len : i+Term])  

    return torch.FloatTensor(np.array(x_seq)).unsqueeze(-1), torch.FloatTensor(np.array(y_seq)).unsqueeze(-1)


def inverse_standard(y, output, scaler):

    y = (y * scaler.scale_) + scaler.mean_
    output = (output * scaler.scale_) + scaler.mean_

    return y.cpu(), output.cpu()

            
            
            
            
            
            
            
            
            

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from skimage.metrics import structural_similarity
from torchmetrics import MeanAbsolutePercentageError


def _mape(pred, target):
    return MeanAbsolutePercentageError()(pred.detach().cpu().reshape(-1), target.detach().cpu().reshape(-1)) 

def _mse(outputs, targets):
    return mean_squared_error(outputs.detach().cpu().numpy().reshape(-1), targets.detach().cpu().numpy().reshape(-1))

def _mae(outputs, targets):
    return mean_absolute_error(outputs.detach().cpu().numpy().reshape(-1), targets.detach().cpu().numpy().reshape(-1))

def _PSNR(pred, true):
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    mse = np.mean((np.uint8(pred * 255)-np.uint8(true * 255))**2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

def _SSIM(predictions, targets, n_features):
    
    targets = targets.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()

    batch_size = predictions.shape[0]
    Seq_len = predictions.shape[1]

    ssim = 0
    
    if n_features == 1:
    
        for batch in range(batch_size):
            for frame in range(Seq_len):
                ssim += structural_similarity(targets[batch, frame].squeeze(), 
                                              predictions[batch, frame].squeeze(),
                                              data_range=predictions[batch, frame].squeeze().max() - predictions[batch, frame].squeeze().min())
    else:
    
        for batch in range(batch_size):
            for frame in range(Seq_len):
                ssim += structural_similarity(targets[batch, frame], predictions[batch, frame],
                                              channel_axis=2, data_range=predictions[batch, frame].squeeze().max() - predictions[batch, frame].squeeze().min())
    ssim /= (batch_size * Seq_len)
    
    return ssim


def metric(outputs, actuals, n_features):
    
    mse = _mse(outputs, actuals)
    mae = _mae(outputs, actuals)
    psnr = _PSNR(outputs, actuals)
    ssim = _SSIM(outputs, actuals, n_features)
    
    return mse, mae, psnr, ssim


#%%


def MSE(pred, true):
    return np.mean((pred - true) ** 2, axis=(0, 1)).sum()

def MAE(pred, true):
    return np.mean(np.abs(pred - true), axis=(0, 1)).sum()

def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255) - np.uint8(true * 255)) ** 2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

def compute_metrics(predictions, targets):
    
    targets = targets.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()

    batch_size = predictions.shape[0]
    Seq_len = predictions.shape[1]

    ssim = 0

    for batch in range(batch_size):
        for frame in range(Seq_len):
            ssim += structural_similarity(targets[batch, frame].squeeze(), 
                                          predictions[batch, frame].squeeze(),
                                          data_range=predictions[batch, frame].squeeze().max() - predictions[batch, frame].squeeze().min())

    ssim /= (batch_size * Seq_len)

    mse = MSE(predictions, targets)
    mae = MAE(predictions, targets)
    psnr = PSNR(predictions, targets)

    return mse, mae, ssim, psnr
    
    
    
    
import os, math
import numpy as np
import scipy.signal
import torch

from piqa.lpips import LPIPS
from piqa.ssim import SSIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssim_model = None
lpips_model = None

def calc_psnr(pred_numpy, gt_numpy):
    pred = torch.tensor(pred_numpy)
    gt = torch.tensor(gt_numpy)
    pred = torch.clip(pred, 0, 1)
    gt = torch.clip(gt, 0, 1)
    mse = torch.mean((pred - gt) ** 2)
    psnr = -10.0 * torch.log(mse) / np.log(10)
    return psnr.cpu().numpy()

def calc_ssim(pred_numpy, gt_numpy):
    global ssim_model
    
    pred = torch.tensor(pred_numpy).to(device)
    gt = torch.tensor(gt_numpy).to(device)
    pred = torch.clip(pred, 0, 1)
    gt = torch.clip(gt, 0, 1)
    pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
    gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
    
    if ssim_model is None:
        ssim_model = SSIM().to(device)
    
    ssim = ssim_model(pred, gt)
    return ssim.detach().cpu().numpy()

def calc_lpips(pred_numpy, gt_numpy):
    global lpips_model
    
    pred = torch.tensor(pred_numpy).to(device)
    gt = torch.tensor(gt_numpy).to(device)
    pred = torch.clip(pred, 0, 1)
    gt = torch.clip(gt, 0, 1)
    pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
    gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
    
    if lpips_model is None:
        lpips_model = LPIPS(network="vgg").to(device)
    
    lpips = lpips_model(pred, gt)
    return lpips.detach().cpu().numpy()


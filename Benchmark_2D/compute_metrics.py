
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import glob
import math
from math import exp
from PIL import Image
# from DataSplit import DataSplit

######################################
## Check MIN & MAX values of images ##
######################################
# def main():
    # # load input data
    # data_root = '/storage/connectome/GANBERT/data/sample/sample_b0_input_ver'
    # test_csv = pd.read_csv('/scratch/connectome/conmaster/Projects/Image_Translation/data_processing/sample_test.csv', header=None)
    # print(len(test_csv))
    #
    # test_data = DataSplit(data_csv=test_csv, data_dir=data_root, do_transform=True)
    # data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=16, pin_memory=False)
    #
    # # Check min & max pixel value of input image
    # s_min = d_min = 10000000
    # s_max = d_max = -10000000
    # for i, data in enumerate(data_loader_test):
    #     struct = data['t1']
    #     struct_min = torch.min(struct)
    #     struct_max = torch.max(struct)
    #     print(struct_min, struct_max)
    #     if s_min > struct_min:
    #         s_min = struct_min
    #     if s_max < struct_max:
    #         s_max = struct_max
    #
    #     dwi = data['dwi']
    #     dwi_min = torch.min(dwi)
    #     dwi_max = torch.max(dwi)
    #     print(dwi_min, dwi_max)
    #     if d_min > dwi_min:
    #         d_min = dwi_min
    #     if d_max < dwi_max:
    #         d_max = dwi_max
    #
    # print("----->", s_min, s_max, d_min, d_max) # -1. & 1.

    # load output data
    # output_dir = '/scratch/connectome/conmaster/Pycharm_projects/3D_brain2brain_GAN/Benchmark_2D/Generated_images/b0_input_wgangp/Test'
    # #
    # fake_images = []
    # real_images = []
    # for img in glob.glob(output_dir+'/*fake.png'):
    #     fake_images.append(np.array(Image.open(img)))
    # for img in glob.glob(output_dir+'/*real.png'):
    #     real_images.append(np.array(Image.open(img)))
    #
    # # Check min & max pixel value of output image
    # f_min = r_min = 10000000
    # f_max = r_max = -10000000
    # for fake in fake_images:
    #     fake_min = np.min(fake)
    #     fake_max = np.max(fake)
    #     print(fake_min, fake_max)
    #     if f_min > fake_min:
    #         f_min = fake_min
    #     if f_max < fake_max:
    #         f_max = fake_max
    #
    # for real in real_images:
    #     real_min = np.min(real)
    #     real_max = np.max(real)
    #     print(real_min, real_max)
    #     if r_min > real_min:
    #         r_min = real_min
    #     if r_max < real_max:
    #         r_max = real_max
    #
    # print("----->", f_min, f_max, r_min, r_max)

######################################
## Calculate metrics : PSNR, SSIM ##
######################################
def PSNR(real_img, fake_img):
    mse = np.mean((real_img - fake_img) ** 2)
    # real_img == fake_img : PSNR 정의될 수 X
    if mse == 0:
        return 100
    # assert np.min(real_img) == np.min(fake_img)
    # assert np.max(real_img) == np.max(fake_img)
    pixel_max = np.max(real_img) - np.min(real_img)
    psnr = 10 * math.log10(pixel_max ** 2 / mse)
    return psnr

def MAE_MSE(real_img, fake_img):
    pix_abs = 0
    pix_sqrt = 0
    for i in range(real_img.shape[0]):
        for j in range(real_img.shape[1]):
            pix_abs += torch.abs(real_img[i, j] - fake_img[i, j])
            pix_sqrt += math.sqrt(real_img[i, j] - fake_img[i, j])
    mae = pix_abs / (real_img.shape[0] * real_img.shape[1])
    mse = pix_sqrt / (real_img.shape[0] * real_img.shape[1])
    return mae, mse

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0)
    print(_1D_window.shape, _2D_window.shape)
    window = Variable(_2D_window.expand(channel, window_size, window_size).contiguous())
    return window

def SSIM(real_img, fake_img, size_average=True):
    real_img = torch.from_numpy(real_img)
    fake_img = torch.from_numpy(fake_img)
    # real_img shape: 1x64x64
    channel = real_img.shape[0]
    window_size = 5
    window = create_window(window_size, channel)

    mu1 = F.conv2d(real_img, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(fake_img, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(real_img * real_img, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(fake_img * fake_img, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(real_img * fake_img, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
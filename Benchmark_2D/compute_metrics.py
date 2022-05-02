
import torch
import pandas as pd
import numpy as np
import glob
from PIL import Image

from DataSplit import DataSplit

######################################
## Check MIN & MAX values of images ##
######################################
# load input data
data_root = '/storage/connectome/GANBERT/data/sample/sample_b0_input_ver'
test_csv = pd.read_csv('/scratch/connectome/conmaster/Projects/Image_Translation/data_processing/sample_test.csv', header=None)
print(len(test_csv))

test_data = DataSplit(data_csv=test_csv, data_dir=data_root, do_transform=True)
data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=16, pin_memory=False)

# Check min & max pixel value of input image
s_min = d_min = 10000000
s_max = d_max = -10000000
for i, data in enumerate(data_loader_test):
    struct = data['t1']
    struct_min = torch.min(struct)
    struct_max = torch.max(struct)
    print(struct_min, struct_max)
    if s_min > struct_min:
        s_min = struct_min
    if s_max < struct_max:
        s_max = struct_max

    dwi = data['dwi']
    dwi_min = torch.min(dwi)
    dwi_max = torch.max(dwi)
    print(dwi_min, dwi_max)
    if d_min > dwi_min:
        d_min = dwi_min
    if d_max < dwi_max:
        d_max = dwi_max

print("----->", s_min, s_max, d_min, d_max)

# load output data
# output_dir = '/scratch/connectome/conmaster/Pycharm_projects/3D_brain2brain_GAN/Benchmark_2D/Generated_images/b0_input_wgangp/Val'
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
    assert np.min(real_img) == np.min(fake_img) and np.max(real_img) == np.max(fake_img)
    pixel_max = np.max(real_img) - np.min(real_img)
    psnr = 10 * log10(pixel_max ** 2 / mse)
    return psnr

def SSIM()
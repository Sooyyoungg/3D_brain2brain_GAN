import torch
from numpy import inf
import torch.nn as nn
from PIL import Image
import numpy as np
import glob
import imageio

# rand = torch.randn(16, 100)
# print(rand.shape)
# lab = torch.randint(0, 10, (16,))
# print(lab.shape, lab)
# emb = nn.Embedding(10, 10)
# lab_emb = emb(lab)
# print(lab_emb.shape)
# conct = torch.cat((lab_emb, rand), -1)
# print(conct.shape)

# print(-inf + inf)
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# torch.backends.cudnn.enabled = True
# print(torch.backends.cudnn.enabled)
#
# for i in range(1, 10):
#     print(i)

# from DCGAN import *
# from DCGAN.model import GAN_3D
# print(GAN_3D.fake_dwi.shape)

# output_dir = '/scratch/connectome/conmaster/Pycharm_projects/3D_brain2brain_GAN/Benchmark_2D/Generated_images/b0_input_wgangp/Test'
# # Benchmark_0001_0016_fake.png / Benchmark_0001_0016_real.png
# print(glob.glob(output_dir+'/*.png'))
# img = Image.open(output_dir+'/Benchmark_0001_0016_fake.png')
# img = np.array(img)
# print(type(img))

# f_min = r_min = 10000000
# print(f_min, r_min)

# from Benchmark_2D.compute_metrics import PSNR, SSIM
# output_root = '/scratch/connectome/conmaster/Pycharm_projects/3D_brain2brain_GAN/Benchmark_2D/Generated_images/b0_input_wgangp/Val'
# total_p = []
# ll = ['0050', '0100', '0150', '0200', '0250', '0300', '0350']
# for i in range(7):
#     r_dwi = imageio.imread(output_root+'/Benchmark_0300_{}_real.png'.format(ll[i]))
#     g_dwi = imageio.imread(output_root+'/Benchmark_0300_{}_fake.png'.format(ll[i]))
#     psnr = PSNR(r_dwi, g_dwi)
#     total_p.append(psnr)
#     print('{}th PSNR: {}'.format(i+1, psnr))
# total_p = np.array(total_p)
# print(np.mean(total_p))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

input = torch.Tensor(np.array([[[ [1,1,1,0,0], [0,1,1,1,0], [0,0,1,1,1], [0,0,1,1,0], [0,1,1,0,0] ]]]))
filter = torch.Tensor(np.array([[[ [1,0,1], [0,1,0], [1,0,1] ]]]))
print(input.shape, filter.shape)
input = Variable(input, requires_grad=True)
filter = Variable(filter)
out = F.conv2d(input, filter)
print(out.shape)

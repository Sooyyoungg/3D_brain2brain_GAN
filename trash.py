import torch
from numpy import inf
import torch.nn as nn
from PIL import Image
import numpy as np
import glob

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

f_min = r_min = 10000000
print(f_min, r_min)

import numpy as np
import pandas as pd
from monai.transforms import ScaleIntensity, NormalizeIntensity
from torchvision import transforms
import imageio

data_root = '/storage/connectome/GANBERT/data/sample/sample_b0_input_ver'
train_csv = pd.read_csv('/scratch/connectome/conmaster/Projects/Image_Translation/data_processing/sample_train.csv', header=None)
# print(train_csv.shape)
sub = train_csv.iloc[5, 1]
print(sub)

## Data Loader
t1 = np.load(data_root+'/'+sub+'.T1.npy')
b0 = np.load(data_root+'/'+sub+'.b0.npy')
dwi = np.load(data_root+'/'+sub+'.dwi.npy')

# print(t1.shape)         (64, 64, 64)
# print(b0.shape)         (64, 64, 64, 7)
# print(dwi.shape)        (64, 64, 64, 96)

## Min & Max pixel intensity
# input original data
print(np.min(t1), np.max(t1))
print(np.min(b0[:,:,:,0]), np.max(b0[:,:,:,0]))
print(np.min(dwi), np.max(dwi))

# normalization & scaling input data
normal_transform = NormalizeIntensity(subtrahend=0.5, divisor=0.5, nonzero=False)
scale_transform = ScaleIntensity(minv=-1.0, maxv=1.0)
transform = transforms.Compose([normal_transform, scale_transform])

t1 = transform(t1)
print(np.min(t1), np.max(t1))   # [-1, 1]

# output data
output_root = '/scratch/connectome/conmaster/Pycharm_projects/3D_brain2brain_GAN/Benchmark_2D/Generated_images/b0_input_wgangp/Train'
r_dwi = imageio.imread(output_root+'/Benchmark_0001_0016_real.png')
g_dwi = imageio.imread(output_root+'/Benchmark_0001_0016_fake.png')

print(np.min(r_dwi), np.max(r_dwi))    # [0, 255]
print(np.min(g_dwi), np.max(g_dwi))    # [0, 255]

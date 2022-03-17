import os
import pandas as pd
from torch.utils.data import DataLoader
import torch
import numpy
from monai.transforms import ScaleIntensity, NormalizeIntensity
from torchvision import transforms

from Config import Config
from DataSplit import DataSplit
from model import GAN_3D

print("<<<<<<<<<<<<<DCGAN model>>>>>>>>>>>>>")

device = torch.device('cuda:'+str(Config.gpu[0]) if torch.cuda.is_available() else 'cpu')
print(device)

os.makedirs("Generated_images", exist_ok=True)

### Data Loader
config = Config()
train_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_train.csv', header=None)
val_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_val.csv', header=None)
test_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_test.csv', header=None)

# sample subjects data: 128 18 36
print(len(train_csv), len(val_csv), len(test_csv))

# split
train_data = DataSplit(data_csv=train_csv, data_dir=config.data_dir, do_transform=True)
val_data = DataSplit(data_csv=val_csv, data_dir=config.data_dir, do_transform=True)
# sub, st, dwi = train_data.__getitem__(3)

# 13184 1854
# print(train_data.__len__(), val_data.__len__())

# load
data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=False, num_workers=16, pin_memory=False)
data_loader_val = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=16, pin_memory=False)

# 412 58
print(len(data_loader_train), len(data_loader_val))

# train_iter = iter(data_loader_train)
# sub, st, dwi = train_iter.next()
# print(st.size(), dwi.size())  #torch.Size([32, 1, 64, 64, 64]) torch.Size([32, 1, 64, 64, 64]) torch.Size([32, 4])

### model
model = GAN_3D([data_loader_train, data_loader_val], config)
model.train()
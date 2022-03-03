import os
import pandas as pd
from torch.utils.data import DataLoader
import torch

from Config import Config
from DataSplit import DataSplit
from model import I2I_cGAN

print("<<<<<<<<<<<<<cGAN model>>>>>>>>>>>>>")

device = torch.device('cuda:'+str(Config.gpu[0]) if torch.cuda.is_available() else 'cpu')
print(device)

os.makedirs("Generated_images", exist_ok=True)

### Data Loader
config = Config()
train_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_train.csv', header=None)
val_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_val.csv', header=None)
test_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_test.csv', header=None)

train_N = len(train_csv)
val_N = len(val_csv)
test_N = len(test_csv)
print(train_N, val_N, test_N)

# split
train_data = DataSplit(data_csv=train_csv, data_dir=config.data_dir, do_transform=None)
val_data = DataSplit(data_csv=val_csv, data_dir=config.data_dir, do_transform=None)

# load
data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=False, num_workers=16, pin_memory=False)
data_loader_val = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=16, pin_memory=False)

# 1648 232
print(len(data_loader_train), len(data_loader_val))

### model
model = I2I_cGAN([data_loader_train, data_loader_val], config)
model.train()
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch

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

train_N = len(train_csv)
val_N = len(val_csv)
test_N = len(test_csv)
# sample subjects data: 128 18 36
print(train_N, val_N, test_N)

# split
train_data = DataSplit(data_csv=train_csv, data_dir=config.data_dir, do_transform=True)
val_data = DataSplit(data_csv=val_csv, data_dir=config.data_dir, do_transform=True)
#s, d, g = train_data.__getitem__(1)
#print(torch.min(s), torch.max(s))
#print(torch.min(d), torch.max(d))
#print(d.shape, g.shape)

# 13184 1854
#print(train_data.__len__(), val_data.__len__())

# load
data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False)
data_loader_val = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False)

# 412 58
print(len(data_loader_train), len(data_loader_val))

#for index, (struct, dwi, grad) in enumerate(data_loader_train):
    # torch.Size([16, 64, 64, 64]) torch.Size([16, 64, 64, 64]) torch.Size([16, 4])
#    print(index, struct.shape, dwi.shape, grad.shape)
#train_iter = iter(data_loader_train)
#st, dwi, grad = train_iter.next()
#print(type(st))   # <class 'torch.Tensor'>
#print(st.size(), dwi.size(), grad.size())  #torch.Size([16, 64, 64, 64]) torch.Size([16, 64, 64, 64]) torch.Size([16, 4])

### model
model = GAN_3D([data_loader_train, data_loader_val], config)
model.train()
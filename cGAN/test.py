### Basic
import argparse
import os
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from Config import Config
from DataSplit import DataSplit
from model import I2I_cGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

os.makedirs("Generated_images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument('-a', '--attribute', type=str, help='Specify category for training.')
parser.add_argument('-g', '--gpu', default=[0], nargs='+', type=int, help='Specify GPU ids.')
parser.add_argument('-r', '--restore', default=None, action='store', type=int, help='Specify checkpoint id to restore.')
parser.add_argument('-m', '--mode', default='test', type=str, choices=['train', 'test'])

args = parser.parse_args()
print(args)

### Data Loader
config = Config()
test_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/preprocessing/sample_code/QC/qc_test.csv', header=None)
test_N = len(test_csv)
# 1225 735 245 245 > sample data
#print(train_N, val_N, test_N)

# split
test_data = DataSplit(data_csv=test_csv, data_dir=config.data_dir, transform=False)
data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False)

### model
model = I2I_cGAN(args, config, data_loader_test, config.epoch)
model.test()
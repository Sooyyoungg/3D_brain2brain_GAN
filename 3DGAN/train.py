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
from model import GAN_3D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

os.makedirs("Generated_images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
"""parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")"""
parser.add_argument('-a', '--attribute', type=str, help='Specify category for training.')
parser.add_argument('-g', '--gpu', default=[0], nargs='+', type=int, help='Specify GPU ids.')
parser.add_argument('-r', '--restore', default=None, action='store', type=int, help='Specify checkpoint id to restore.')
parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'test'])

args = parser.parse_args()
print(args)

### Data Loader
config = Config()
train_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/preprocessing/sample_code/qc_train.csv', header=None)
val_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/preprocessing/sample_code/qc_val.csv', header=None)
test_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/preprocessing/sample_code/qc_test.csv', header=None)

train_N = len(train_csv)
val_N = len(val_csv)
test_N = len(test_csv)
# 1225 735 245 245 > sample data
print(train_N, val_N, test_N)

# split
train_data = DataSplit(data_csv=train_csv, data_dir=config.data_dir, transform=None)
val_data = DataSplit(data_csv=val_csv, data_dir=config.data_dir, transform=None)
test_data = DataSplit(data_csv=test_csv, data_dir=config.data_dir, transform=None)
#s = train_data.__getitem__(0)

# load
data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False)
data_loader_val = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False)
data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False)

for index, struct , dwi in enumerate(data_loader_train):
    print(index, struct.shape, dwi.shape)
#train_iter = iter(data_loader_train)vv
#st = train_iter.next()
#print(type(st))   # <class 'torch.Tensor'>
#print(st.size())  # torch.Size([64, 2, 256, 256, 256])
print("Data Ready !!!")

### model
#model = GAN_3D(args, [data_loader_train, data_loader_val], config, config.epoch)
#model.train()

"""### Generator & Discriminator
# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function
adversarial_loss = torch.nn.BCELoss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


### Training
Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor

for epoch in range(args.n_epochs):
    for i, (img, _) in enumerate(data_loader_train):
        print(struct.shape)  # torch.Size([64, 1, 28, 28])
        print(dwi.shape)

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        """""" Generator """"""
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        """""" Discriminator """"""
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
              % (epoch, opt.n_epochs, i, len(data_loader_train), d_loss.item(), g_loss.item()))

        batches_done = epoch * len(data_loader_train) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "T1_T2_generated_images/%d.png" % batches_done, nrow=5, normalize=True)

        """""" Validation """"""
        if i % 10 == 0:
            with torch.no_grad():
                val_loss = 0.0
                for v, (v_imgs, _) in enumerate(data_loader_val):
                    val_imgs = Variable(v_imgs.type(Tensor))
                    g_val = generator(val_imgs)


torch.save({
    'epoch': opt.n_epochs,
    'batch': opt.batch_size,
    'model_G_state_dict': generator.state_dict(),
    'model_D_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict()}, 'model_weights.pth')

### Testing
# with torch.no_grad():
"""
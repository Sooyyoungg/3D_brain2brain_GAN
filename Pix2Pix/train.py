import time
import pandas as pd
import torch
import numpy
from torch.utils.data import DataLoader

from Config import Config
from DataSplit import DataSplit
from model import Pix2Pix

### Data Loader
config = Config()
train_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_train.csv', header=None)
val_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_val.csv', header=None)

# sample subjects data: 128 18 36
print(len(train_csv), len(val_csv))

# split
train_data = DataSplit(data_csv=train_csv, data_dir=config.data_dir, do_transform=True)
val_data = DataSplit(data_csv=val_csv, data_dir=config.data_dir, do_transform=True)

# load
data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=False, num_workers=16, pin_memory=False)
data_loader_val = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=16, pin_memory=False)
print(len(data_loader_train), len(data_loader_val))

### Model Loader
model = Pix2Pix()

### Training
total_iters = 0
train_start_time = time.time()
for epoch in range(config.n_epoch):
    ep_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(data_loader_train):
        total_iters += config.batch_size
        epoch_iter += config.batch_size

        it_start_time = time.time()
        train_dict = model.train(data)

        # loss
        G_loss = train_dict['G_loss']
        D_loss = train_dict['D_loss']

        it_finish_time = time.time()
        it_time = it_finish_time - it_start_time
        print('[Time %.3fs | epoch: %d | it: %d | total_it: %d | G_loss: %.7fs | D_loss: %.7fs').format(it_time, epoch, epoch_iter, total_iters, G_loss, D_loss)

        # Validation
        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                data_val = next(iter(data_loader_val))
                val_dict = model.val(data_val)

                # Save generated image - Training data
                plt.imsave(
                    os.path.join(config["img_dir"], 'Train', '{:04d}_{:04d}_real.png'.format(epoch, it + 1)),
                    train_dict['dwi'][:, :], cmap='gray')
                plt.imsave(
                    os.path.join(config["img_dir"], 'Train', '{:04d}_{:04d}_fake.png'.format(epoch, it + 1)),
                    train_dict['pred'][:, :], cmap='gray')

                # Save generated image - Validation data
                plt.imsave(
                    os.path.join(config["img_dir"], 'Val', '{:04d}_{:04d}_real.png'.format(epoch, it + 1)),
                    val_dict['dwi'][:, :], cmap='gray')
                plt.imsave(
                    os.path.join(config["img_dir"], 'Val', '{:04d}_{:04d}_fake.png'.format(epoch, it + 1)),
                    val_dict['pred'][:, :], cmap='gray')

    # save model
    if (epoch + 1) % config['snapshot_save_iter'] == 0:
        trainer.save(log_dir, epoch, iterations)
    if (epoch + 1) % config['latest_save_iter'] == 0:
        trainer.save(log_dir, -1)
        with open(log_dir + '/latest_log.txt', 'w') as f:
            f.writelines('%d, %d'%(epoch, iterations))

    ep_finish_time = time.time()
    epoch_time = ep_finish_time - ep_start_time
    print("%dth epoch time: %.5fs").format(epoch, epoch_time)

train_finish_time = time.time()
train_time = train_finish_time - train_start_time
print("Finish Training! Total time: %.5fs").format(train_time)
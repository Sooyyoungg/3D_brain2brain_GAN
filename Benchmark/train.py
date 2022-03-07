import sys, os
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
from nilearn import plotting

sys.path.append('..')
sys.path.append('.')
import argparse
import pandas as pd
from trainer import dwi_Trainer
from time import time
import os
import tensorboardX
import shutil
# from data.dwi_loader import h5Loader
import torch.utils.data
from utils.visualization import tensorboard_vis
from utils.utilization import mkdirs, convert, get_config
from DataSplit import DataSplit
import matplotlib
matplotlib.use('Agg')

print("<<<<<<<<<<<<<Benchmark model>>>>>>>>>>>>>")

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='smri2dwi.yaml', help='Path to the config file.')
parser.add_argument('--data_root', type=str, default='', help='Path to the data, if None, get from config files')
parser.add_argument("--resume", type=int, default=0)
opts = parser.parse_args()

### Load experiment setting
config = get_config(opts.config)
n_epochs = config['n_epoch']
n_iterations = config['n_iterations']
display_size = config['display_size']
batch_size = config['batch_size']

### Setup model and data loader
trainer = dwi_Trainer(config)
trainer.to(trainer.device)

n_dwi = config['n_dwi']
load_t1 = config['multimodal_t1']

train_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_train.csv', header=None)
val_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_val.csv', header=None)
test_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_test.csv', header=None)

train_N = len(train_csv)
val_N = len(val_csv)
test_N = len(test_csv)
# sample data: 128 18 36
print(train_N, val_N, test_N)

# split
train_data = DataSplit(data_csv=train_csv, data_dir=config['data_root'], do_transform=True)
val_data = DataSplit(data_csv=val_csv, data_dir=config['data_root'], do_transform=True)

# load
data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=16, pin_memory=False)
data_loader_val = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'], shuffle=True, num_workers=16, pin_memory=False)

### Setup logger and output folders
log_dir = config['log_dir']
if not os.path.exists(log_dir):
    print('* Creating log directory: ' + log_dir)
    mkdirs(log_dir)
print('* Logs will be saved under: ' + log_dir)
train_writer = tensorboardX.SummaryWriter(log_dir)
print('* Creating tensorboard summary writer ...')
if not os.path.exists(os.path.join(log_dir, 'config.yaml')):
    shutil.copy(opts.config, os.path.join(log_dir, 'config.yaml')) # copy config file to output folder

## Load model
if config['pretrained'] != '':
    trainer.resume(config['pretrained'])

load_epoch = int(opts.resume)
iterations = 0
if opts.resume > 0:
    # 가장 마지막으로 학습된 log 기록을 통해서 어디까지 학습됐는지 불러오기
    with open(log_dir + '/latest_log.txt', 'r') as f:
        x = f.readlines()[0]
        load_epoch, iterations = int(x.split(',')[0]), int(x.split(',')[1])
        if load_epoch == -1:
            load_epoch = int(iterations/ len(data_loader_train))
        if iterations == -1:
            iterations = load_epoch*len(data_loader_train)

    # 학습된 부분까지의 model 불러오기
    load_suffix = 'epoch%d.pt'%load_epoch
    if not os.path.exists(log_dir + '/gen_' + load_suffix):
        load_suffix = 'latest.pt'
    if not os.path.exists(log_dir + '/gen_latest.pt'):
        load_suffix = 'best.pt'
    print('* Resume training from {}'.format(load_suffix))

    state_dict = torch.load(log_dir + '/gen_'+load_suffix)
    trainer.gen_a.load_state_dict(state_dict['a'])

    opt_dict = torch.load(log_dir + '/opt_' + load_suffix)
    trainer.gen_opt.load_state_dict(opt_dict['gen'])

    if trainer.gan_w > 0:
        state_dict = torch.load(log_dir + '/dis_'+load_suffix)
        trainer.dis.load_state_dict(state_dict['dis'])
        trainer.dis_opt.load_state_dict(opt_dict['dis'])

## Start training
print('* Training from epoch %d'%load_epoch)
print('---------------------------------------------------------------------')
print('lambda L1: %.2f, gan: %.2f'%(trainer.l1_w, trainer.gan_w))
print('---------------------------------------------------------------------')
best_train_loss, best_val_loss = 999, 999
epoch = load_epoch
start = time()
while epoch < n_epochs or iterations < n_iterations:
    epoch += 1
    for it, data in enumerate(data_loader_train):
        iterations = it + epoch*len(data_loader_train)
        # 학습 시간 출력
        start = time()
        train_dict = trainer.update(data, n_dwi, iterations)     # : (64, 64, 64)
        end = time()
        update_t = end - start

        # Loss 값 계산
        ldwi = trainer.loss_dwi.item()
        lg, ld = trainer.loss_g.item(), trainer.loss_d.item()
        loss_print = ''
        loss_print += ' Loss_dwi: %.4f'%ldwi if trainer.l1_w>0 else ''
        loss_print += ' Loss_g: %.4f, Loss_d: %.4f'%(lg, ld) if trainer.gan_w > 0 else ''
        print('[Time %.3fs/it %d: %d/%d, Iter: %d (lr:%.5f)] '%(update_t, epoch, it, len(data_loader_train),
                                                             iterations, trainer.gen_opt.param_groups[0]['lr']) + loss_print)
        # Update learning rate
        trainer.update_learning_rate()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            if ldwi > 0: train_writer.add_scalar('loss_dwi', trainer.loss_dwi, iterations)
            if ld > 0: train_writer.add_scalar('loss_d', trainer.loss_d, iterations)
            if lg > 0: train_writer.add_scalar('loss_g', trainer.loss_g, iterations)

        # Write images
        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                data_test = next(iter(data_loader_val))
                test_ret = trainer.sample(data_test)
                # print(imgs_titles)
                # cmaps = ['jet' if 'seg' in i else 'gist_gray' for i in imgs_titles]
                # writer = tensorboard_vis(summarywriter=train_writer, step=iterations, board_name='val/',
                #                          num_row=2, img_list=imgs_vis, cmaps=cmaps,
                #                          titles=imgs_titles, resize=True)
                #
                # imgs_vis = [train_dict[k] for k in train_dict.keys()]
                # imgs_titles = list(train_dict.keys())
                # cmaps = ['jet' if 'seg' in i else 'gist_gray' for i in imgs_titles]
                # writer = tensorboard_vis(summarywriter=train_writer, step=iterations, board_name='train/',
                #                          num_row=2, img_list=imgs_vis, cmaps=cmaps,
                #                          titles=imgs_titles, resize=True)

                # (64, 64, 64) (64, 64, 64) (64, 64, 64) (4,)
                # print(train_dict['t1'].shape, train_dict['dwi'].shape, train_dict['pred'].shape, train_dict['grad'].shape)

                # Save generated image - Training data
                plt.imsave(os.path.join(config["img_dir"], 'Train', 'Benchmark_{:04d}_{:04d}_real.png'.format(epoch, it+1)), train_dict['dwi'][:,:,32], cmap='gray')
                plt.imsave(os.path.join(config["img_dir"], 'Train', 'Benchmark_{:04d}_{:04d}_fake.png'.format(epoch, it+1)), train_dict['pred'][:,:,32], cmap='gray')

                # Save generated image - Testing data
                plt.imsave(os.path.join(config["img_dir"], 'Test', 'Benchmark_{:04d}_{:04d}_real.png'.format(epoch, it + 1)), test_ret['dwi'][:, :, 32], cmap='gray')
                plt.imsave(os.path.join(config["img_dir"], 'Test', 'Benchmark_{:04d}_{:04d}_fake.png'.format(epoch, it + 1)), test_ret['pred'][:, :, 32], cmap='gray')

                # Visualize generated image
                # feat = np.squeeze((0.5 * train_dict['dwi'] + 0.5))
                # feat = nib.Nifti1Image(feat, affine=np.eye(4))
                # plotting.plot_anat(feat, title="Real_imgs", cut_coords=(32, 32, 32))
                # plotting.show()
                #
                # feat_f = np.squeeze((0.5 * train_dict['pred'] + 0.5))
                # feat_f = nib.Nifti1Image(feat_f, affine=np.eye(4))
                # plotting.plot_anat(feat_f, title="Generated_imgs", cut_coords=(32, 32, 32))
                # plotting.show()


    # Save network weights
    if (epoch + 1) % config['snapshot_save_iter'] == 0:
        trainer.save(log_dir, epoch, iterations)
    if (epoch + 1) % config['latest_save_iter'] == 0:
        trainer.save(log_dir, -1)
        with open(log_dir + '/latest_log.txt', 'w') as f:
            f.writelines('%d, %d'%(epoch, iterations))
end = time()
print('Training finished in {}, {} epochs, {} iterations'.format(convert(end-start), epoch, iterations))


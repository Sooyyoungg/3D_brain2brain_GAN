import argparse

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from trainer import dwi_Trainer
import os
from utils.utilization import get_config
from DataSplit import DataSplit
from compute_metrics import PSNR, SSIM, MAE_MSE

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='smri2dwi.yaml', help='Path to the config file.')
parser.add_argument('--data_root', type=str, default='', help='Path to the data, if None, get from config files')
opts = parser.parse_args()
config = get_config(opts.config)

## Data loader
data_root = '/storage/connectome/GANBERT/data/sample/sample_b0_input_ver'

test_csv = pd.read_csv('/scratch/connectome/conmaster/Projects/Image_Translation/data_processing/sample_test.csv', header=None)
print(len(test_csv))
# 36 x 96 = 3456

test_data = DataSplit(data_csv=test_csv, data_dir=data_root, do_transform=True)
data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=16, pin_memory=False)

## Load model
trainer = dwi_Trainer(config)
trainer.to(trainer.device)

if config['pretrained'] != '':
    trainer.resume(config['pretrained'])

log_dir = config['log_dir']
load_epoch = 0
iterations = 0

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

state_dict = torch.load(log_dir + '/gen_'+load_suffix, map_location=trainer.device)
trainer.gen_a.load_state_dict(state_dict['a'])

opt_dict = torch.load(log_dir + '/opt_' + load_suffix, map_location=trainer.device)
trainer.gen_opt.load_state_dict(opt_dict['gen'])

## Testing
trainer.eval()
with torch.no_grad():
    print("Testing!!!")
    psnr_total = []
    ssim_total = []
    for i, data in enumerate(data_loader_test):
        test_result = trainer.sample(data)
        # print(np.min(test_result['dwi']), np.max(test_result['dwi']))
        # print(np.min(test_result['pred']), np.max(test_result['pred']))

        # calculate PSNR
        psnr = PSNR(test_result['dwi'], test_result['pred'])
        psnr_total.append(psnr)

        # calculate SSIM
        #ssim = SSIM(test_result['dwi'], test_result['pred'])
        #ssim_total.append(ssim)

        # calculate MAE & MSE
        mae, mse = MAE_MSE(test_result['dwi'], test_result['pred'])
        print('{}th PSNR: {}  |  MAE: {}  |  MSE: {}'.format(i+1, psnr, mae, mse))


        # Save generated image - Testing data
        plt.imsave(os.path.join(config["img_dir"], 'Test', 'Test_{}_real.png'.format(i+1)), test_result['dwi'][:, :], cmap='gray')
        plt.imsave(os.path.join(config["img_dir"], 'Test', 'Test_{}_fake.png'.format(i+1)), test_result['pred'][:, :], cmap='gray')

        #print('{}th PSNR: {}  |  SSIM: {}'.format(i+1, psnr, ssim))
    print('Testing mean result\n PSNR: {}'.format(np.mean(np.array(psnr_total))))
    #print('Testing mean result\n PSNR: {}  |  SSIM: {}'.format(np.mean(np.array(psnr_total)), np.mean(np.array(ssim))))
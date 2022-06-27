import time

import cv2
import torch
import pandas as pd
import numpy as np
from skimage.metrics import mean_squared_error
from torchvision.utils import save_image
import copy as cp

from Config import Config
from DataSplit_test import DataSplit
from model import Pix2Pix


def main():
    config = Config()
    device = torch.device('cuda:{}'.format(config.gpu_ids[0])) if config.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    print(device)

    ## Data Loader
    valid_data = DataSplit(data_list=config.valid_list, data_root=config.valid_root)
    data_loader_valid = torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=True, num_workers=16, pin_memory=False)
    print("Train: ", len(data_loader_valid), "x", 1,"(batch size) =", len(valid_data))

    ## Start Training
    model = Pix2Pix(config)
    model.load_state_dict(torch.load(config.log_dir+'/SEM_best_epoch72_itr14040000_rmse2.812597536459417.pt'))
    model.to(device)
    # model = cp.copy(model)

    print("Start Validation!!")
    tot_itr = 0
    mse = 0
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(data_loader_valid):
            tot_itr += i
            test_dict = model.test(data)

            fake_depth = test_dict['fake_depth']
            sub = test_dict['sub'][0]
            real_depth = data['depth']

            # post-processing & MSE
            f_image = ((fake_depth[0, 0, :, :].detach().cpu().numpy() + 1) / 2) * 255.0
            r_image = ((real_depth[0, 0, :, :].detach().cpu().numpy() + 1) / 2) * 255.0
            mse += mean_squared_error(f_image, r_image)

            # image 저장
            print(i, "th image save")
            cv2.imwrite('{}/{}'.format(config.valid_img_dir,sub), f_image)

    # RMSE
    rmse = (mse / len(data_loader_valid)) ** 0.5
    print("RMSE: ", rmse)

    end_time = time.time()
    print("Testing Time: ", end_time - start_time)

if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import nibabel as nib
from skimage.transform import resize

import matplotlib.pyplot as plt
import matplotlib.image as img

class DataSplit(nn.Module):
    def __init__(self, config, data_list, do_transform=True):
        super(DataSplit, self).__init__()

        self.data_csv = data_list
        self.t1_dir = config.t1_dir
        self.b0_dir = config.b0_dir
        self.dwi_dir = config.dwi_dir
        self.grad_dir = config.grad_dir

        self.do_transform = do_transform
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

        ### Data Concatenate
        self.total_t1 = []
        self.total_b0 = []
        self.total_dwi = []
        self.total_grad = []

        for i in range(len(self.data_csv)):
            sub = self.data_csv.iloc[i][0]
            # print(sub)

            ## T1
            t1 = nib.load(self.t1_dir + '/' + sub + '.T1.nii.gz').get_fdata()  # (140, 140, 140)
            t1 = np.reshape(t1, (1, 140, 140, 140))

            ## B0
            b0_raw = np.load(self.b0_dir + '/' + sub + '.b0.npy')  # (7, 140, 140, 140)
            b0 = np.reshape(b0_raw[0, :, :, :], (1, 140, 140, 140))

            ## DWI
            dwi_total = np.load(self.dwi_dir + '/' + sub + '.dwi.npy')  # (96, 140, 140, 140)

            ## Gradient : change grad file into numpy
            grad_file = open(self.grad_dir + '/' + sub + '.grad.b').read()
            grad_n = np.array(grad_file.split('\n'))
            gg = []
            for i in range(len(grad_n)):
                one_grad = grad_n[i].split(' ')
                gg.append([float(one_grad[0]), float(one_grad[1]), float(one_grad[2]), float(one_grad[3])])
            grad_total = np.array(gg)  # (96, 4)

            t1 = t1[:, :, 70]  # (1, 140, 140)
            b0 = b0[:, :, 70]  # (1, 140, 140)
            for j in range(dwi_total.shape[0]):
                dwi = dwi_total[j, :, :, 70]  # (140, 140)
                grad = grad_total[j]  # (4)

                self.total_t1.append(t1)
                self.total_b0.append(b0)
                self.total_dwi.append(dwi)
                self.total_grad.append(grad)

        self.total_t1 = np.array(self.total_t1)  # (12192, 2, 64, 64)
        self.total_b0 = np.array(self.total_b0)  # (12192, 2, 64, 64)
        self.total_dwi = np.array(self.total_dwi)  # (12192, 64, 64)
        self.total_grad = np.array(self.total_grad)  # (12192, 4)
        print(self.total_t1.shape, self.total_b0.shape, self.total_dwi.shape, self.total_grad.shape)

    def __len__(self):
        return len(self.total_dwi)

    def __getitem__(self, index):
        t1 = self.total_t1[index]
        b0 = self.total_b0[index]
        dwi = self.total_dwi[index]
        grad = self.total_grad[index]

        # Transform
        if self.do_transform is not None:
            t1 = self.transform(t1)
            b0 = self.transform(b0)
            dwi = self.transform(dwi)

        print(t1.shape, torch.min(t1), torch.max(t1))
        print(b0.shape, torch.min(b0), torch.max(b0))
        print(dwi.shape, torch.min(dwi), torch.max(dwi))

        # Reshape
        # struct = struct.reshape((1, 64, 64))
        # dwi = dwi.reshape((1, 64, 64))
        # grad = grad.reshape((1, 4))

        return {"input": struct, "dwi": dwi, "cond": grad}

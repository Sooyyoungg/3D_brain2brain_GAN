import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import nibabel as nib
from monai.transforms import ScaleIntensity, NormalizeIntensity
from torchvision import transforms

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
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
        normal_transform = NormalizeIntensity(subtrahend=0.5, divisor=0.5, nonzero=False)
        scale_transform = ScaleIntensity(minv=-1.0, maxv=1.0)
        self.transform = transforms.Compose([normal_transform, scale_transform, transforms.ToTensor()])

        ### Data Concatenate
        self.subs = []
        self.total_dwi = []
        self.total_grad = []
        for i in range(len(self.data_csv)):
            sub = self.data_csv.iloc[i][1]

            ## Gradient : change grad file into numpy
            grad_file = open(self.grad_dir + '/' + sub + '.grad.b').read()
            grad_n = np.array(grad_file.split('\n'))

            for j in range(grad_n.shape[0]):
                grad_sp = grad_n[j].split(' ')
                grad = []
                for i in range(len(grad_sp)):
                    grad.append(float(grad_sp[0]))
                grad = np.array(grad)
                self.total_grad.append([j, grad])
                self.subs.append(sub)

        # print(len(self.subs))                      # 47910
        self.total_grad = np.array(self.total_grad)  # (47910, 2)

    def __len__(self):
        return len(self.subs)

    def __getitem__(self, index):
        ## Data Load
        sub = self.subs[index]

        """ Data Size
            - T1: (140, 140, 140) -> (140, 140) -> (1, 140, 140) => torch.Size([1, 140, 140])
            - B0: (7, 140, 140, 140) -> (140, 140) -> (1, 140, 140) => torch.Size([1, 140, 140])
            - DWI: (96, 140, 140, 140) -> (140, 140) -> (1, 140, 140) => torch.Size([1, 140, 140])
            - Gradient: (4,) """

        # T1
        t1 = nib.load(self.t1_dir + '/' + sub + '.T1.nii.gz').get_fdata()[:, :, 70]    # (140, 140, 140) -> (140, 140)
        # B0
        b0 = np.load(self.b0_dir + '/' + sub + '.b0.npy')[0, :, :, 70]                 # (7, 140, 140, 140) -> (140, 140)
        # Gradient
        idx, grad = self.total_grad[index]                                             # (4,)
        # DWI
        dwi = np.load(self.dwi_dir + '/' + sub + '.dwi.npy')[idx, :, :, 70]            # (96, 140, 140, 140) -> (140, 140)

        ## Transform
        if self.do_transform is not None:
            t1 = self.transform(t1)     # torch.Size([1, 140, 140])
            b0 = self.transform(b0)     # torch.Size([1, 140, 140])
            dwi = self.transform(dwi)   # torch.Size([1, 140, 140])

        # plt.imshow(b0[0, :, :], cmap='gray')
        # plt.show()

        struct = torch.cat((t1, b0), dim=0) # torch.Size([2, 140, 140])
        grad = np.reshape(grad, (1, 4))

        return {"input": struct, "dwi": dwi, "cond": grad}

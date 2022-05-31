import numpy as np
import torch
from monai.transforms import ScaleIntensity, NormalizeIntensity
from torch.utils.data import Dataset
from torchvision import transforms

class DataSplit(Dataset):
    def __init__(self, data_csv, data_dir, do_transform=True):
        super(DataSplit, self).__init__()

        self.data_csv = data_csv
        self.data_dir = data_dir

        self.do_transform = do_transform
        normal_transform = NormalizeIntensity(subtrahend=0.5, divisor=0.5, nonzero=False)
        scale_transform = ScaleIntensity(minv=-1.0, maxv=1.0)
        self.transform = transforms.Compose([normal_transform, scale_transform, transforms.ToTensor()])

        ### Data Concatenate
        self.total_st = []
        self.total_dwi = []
        self.total_grad = []

        for i in range(len(self.data_csv)):
            sub = self.data_csv.iloc[i][1]

            # Structure & diffusion-weighted image & Gradient
            struct = np.load(self.data_dir + '/' + sub + '.T1.npy')  # (64, 64, 64)
            b0_raw = np.load(self.data_dir + '/' + sub + '.b0.npy')  # (64, 64, 64, 7)
            b0s = np.transpose(b0_raw, (3, 0, 1, 2))  # (7, 64, 64, 64)
            dwi_raw = np.load(self.data_dir + '/' + sub + '.dwi.npy')  # (64, 64, 64, 96)
            dwi_total = np.transpose(dwi_raw, (3, 0, 1, 2))  # (96, 64, 64, 64)
            grad_file = open(self.data_dir + '/' + sub + '.grad.b').read()

            # change grad file into numpy
            grad_list = grad_file.split('\n')
            grad_n = np.array(grad_list)
            gg = []
            for i in range(len(grad_n)):
                one_grad = grad_n[i].split(' ')
                gg.append([float(one_grad[0]), float(one_grad[1]), float(one_grad[2]), float(one_grad[3])])
            grad_total = np.array(gg)  # (96, 4)

            for j in range(dwi_total.shape[0]):
                # input_3D = np.concatenate((struct.reshape((1, 64, 64, 64)), b0), axis=0)  # (8, 64, 64, 64)
                # input = input_3D[:, :, :, 32] # (8, 64, 64)
                t1 = struct[:, :, 32]
                b0 = b0s[0, :, :, 32]
                dwi = dwi_total[j, :, :, 32]  # (64, 64)
                grad = grad_total[j]  # (4)
                self.total_t1.append(t1)
                self.total_b0.append(b0)
                self.total_dwi.append(dwi)
                self.total_grad.append(grad)

        self.total_t1 = np.array(self.total_t1)  # (12288, 64, 64)
        self.total_b0 = np.array(self.total_b0)  # (12288, 64, 64)
        self.total_dwi = np.array(self.total_dwi)  # (12288, 64, 64)
        self.total_grad = np.array(self.total_grad)  # (12288, 4)
        print(self.total_t1.shape, self.total_b0.shape)

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

        # Reshape
        t1 = t1.reshape((1, 64, 64))
        b0 = b0.reshape((1, 64, 64))
        dwi = dwi.reshape((1, 64, 64))
        # grad = grad.reshape((1, 4))

        return {"t1": t1, "b0": b0, "dwi": dwi, "grad": grad}
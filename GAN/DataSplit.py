import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DataSplit(Dataset):
    def __init__(self, data_csv, data_dir, do_transform=True):
        super(DataSplit, self).__init__()

        self.data_csv = data_csv
        self.data_dir = data_dir
        self.do_transform = do_transform

        self.count = 0

    def __len__(self):
        return len(self.data_csv) * 103

    def __getitem__(self, index):
        sub = self.data_csv.iloc[self.count][1]
        #T1 = np.load(self.data_dir + '/' + sub + '.T1.npy')    # (256, 256, 256)
        #T2 = np.load(self.data_dir + '/' + sub + '.T2.npy')    # (256, 256, 256)
        #T1 = T1.reshape((1, 256, 256, 256))
        #T2 = T2.reshape((1, 256, 256, 256))
        #struct = np.concatenate([T1, T2], axis=0)               # (2, 256, 256, 256)

        if index != 0 and index % 103 == 0:
            self.count += 1

        struct = np.load(self.data_dir + '/' + sub + '.T1.npy')     # (64, 64, 64)
        struct = struct.reshape((1, 64, 64, 64))

        dwi_raw = np.load(self.data_dir + '/' + sub + '.dwi.npy')   # (64, 64, 64, 103)
        dwi_total = np.transpose(dwi_raw, (3, 0, 1, 2))             # (103, 64, 64, 64)
        dwi = dwi_total[index % 103, :, :, :]
        dwi = dwi.reshape((1, 64, 64, 64))

        grad_file = open(self.data_dir + '/' + sub + '.grad.b').read()
        # change grad file into numpy
        grad_list = grad_file.split('\n')
        grad_n = np.array(grad_list)
        gg = []
        for i in range(len(grad_n)):
            one_grad = grad_n[i].split(' ')
            gg.append([float(one_grad[0]), float(one_grad[1]), float(one_grad[2]), float(one_grad[3])])
        grad_total = np.array(gg)
        grad = grad_total[index % 103]

        if self.do_transform is not None:
            struct = torch.Tensor(struct)
            dwi = torch.Tensor(dwi)
            grad = torch.Tensor(grad)

        return struct, dwi, grad
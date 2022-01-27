import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class DataSplit(Dataset):
    def __init__(self, data_csv, data_dir, transform=None):
        super(DataSplit, self).__init__()

        self.data_csv = data_csv
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, index):
        sub = self.data_csv.iloc[index][1]

        T1 = np.load(self.data_dir + '/' + sub + '.T1.npy')         # (256, 256, 256)
        dwi_raw = np.load(self.data_dir + '/' + sub + '.dwi.npy')   # (190, 190, 190, 103)
        dwi = np.transpose(dwi_raw, (3, 0, 1, 2))                   # (103, 190, 190, 190)
        grad_file = open(self.data_dir + '/' + sub + '.grad.b').read()

        # change grad file into numpy
        grad_list = grad_file.split('\n')
        grad_n = np.array(grad_list)
        gg = []
        for i in range(len(grad_n)):
            one_grad = grad_n[i].split(' ')
            gg.append([float(one_grad[0]), float(one_grad[1]), float(one_grad[2]), one_grad[3]])
        grad = np.array(gg)

        if self.transform is not None:
            T1 = self.transform(T1)
            dwi = self.transform(dwi)

        return dwi, grad
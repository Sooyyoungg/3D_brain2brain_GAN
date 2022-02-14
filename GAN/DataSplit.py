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
        self.data_num = len(data_csv)

        self.do_transform = do_transform
        normal_transform = NormalizeIntensity(subtrahend=0.5, divisor=0.5, nonzero=False)
        scale_transform = ScaleIntensity(minv=0.0, maxv=1.0)
        self.transform = transforms.Compose([normal_transform, scale_transform, transforms.ToTensor()])

        self.count = 0

    def __len__(self):
        return len(self.data_csv) * 103

    def __getitem__(self, index):
        if index == 0:
            self.count = 0
        if index != 0 and index % self.data_num == 0:
            self.count += 1
        #print(self.count, index)
        try:
            sub = self.data_csv.iloc[self.count][1]
        except:
            print("Error")
            print(self.count, index)

        ### Structure & diffusion-weighted image
        struct = np.load(self.data_dir + '/' + sub + '.T1.npy')     # (64, 64, 64)

        dwi_raw = np.load(self.data_dir + '/' + sub + '.dwi.npy')   # (64, 64, 64, 103)
        dwi_total = np.transpose(dwi_raw, (3, 0, 1, 2))             # (103, 64, 64, 64)
        dwi = dwi_total[index % 103, :, :, :]

        ### Gradient
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

        # random example
        #struct = np.random.random_sample((64, 64, 64))
        #dwi = np.random.random_sample((64, 64, 64))
        ### Transform
        if self.do_transform is not None:
            struct = self.transform(struct)
            dwi = self.transform(dwi)
            #grad = self.transform(grad)

        struct = struct.reshape((1, 64, 64, 64))
        dwi = dwi.reshape((1, 64, 64, 64))

        return struct, dwi, grad
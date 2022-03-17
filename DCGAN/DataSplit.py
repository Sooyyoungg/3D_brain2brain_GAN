import numpy as np
import torch
from monai.transforms import ScaleIntensity, NormalizeIntensity
from torch.utils.data import Dataset
from torchvision import transforms

class DataSplit(Dataset):
    def __init__(self, data_csv, data_dir, do_transform=True):
        self.data_csv = data_csv
        self.data_dir = data_dir

        self.do_transform = do_transform
        normal_transform = NormalizeIntensity(subtrahend=0.5, divisor=0.5, nonzero=False)
        scale_transform = ScaleIntensity(minv=-1.0, maxv=1.0)
        self.transform = transforms.Compose([normal_transform, scale_transform, transforms.ToTensor()])

        ### Data Concatenate
        self.total_st = []
        self.total_dwi = []

        for i in range(len(self.data_csv)):
            sub = self.data_csv.iloc[i][1]

            # Structure & diffusion-weighted image
            struct = np.load(self.data_dir + '/' + sub + '.T1.npy')    # (64, 64, 64)
            dwi_raw = np.load(self.data_dir + '/' + sub + '.dwi.npy')  # (64, 64, 64, 103)
            dwi_total = np.transpose(dwi_raw, (3, 0, 1, 2))            # (103, 64, 64, 64)

            for j in range(103):
                dwi = dwi_total[j,:,:,:]                               # (64, 64, 64)
                self.total_st.append(struct)
                self.total_dwi.append(dwi)

        self.total_st = np.array(self.total_st)                        # (13184, 64, 64, 64)
        self.total_dwi = np.array(self.total_dwi)                      # (13184, 64, 64, 64)

    ### Define functions
    def __len__(self):
        return len(self.total_dwi)

    def __getitem__(self, index):
        struct = self.total_st[index]
        dwi = self.total_dwi[index]

        # Transform
        if self.do_transform is not None:
            struct = self.transform(struct)
            dwi = self.transform(dwi)

        # Reshape
        struct = struct.reshape((1, 64, 64, 64))
        dwi = dwi.reshape((1, 64, 64, 64))

        return struct, dwi
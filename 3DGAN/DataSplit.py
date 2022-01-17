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

        T1 = np.load(self.data_dir + '/' + sub + '.T1.npy')    # (256, 256, 256)
        T2 = np.load(self.data_dir + '/' + sub + '.T2.npy')    # (256, 256, 256)
        T1 = T1.reshape((1, 256, 256, 256))
        T2 = T2.reshape((1, 256, 256, 256))
        struct = np.concatenate([T1, T2], axis=0)               # (2, 256, 256, 256)
        dwi = np.load(self.data_dir + '/' + sub + '.dwi.npy')   # (103, 190, 190, 190)
        print(dwi.shape)
        grad = open(self.data_dir + '/' + sub + '.grad.b', "w")

        if self.transform is not None:
            struct = self.transform(struct)
            dwi = self.transform(dwi)

        return struct, dwi #, grad
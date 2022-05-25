

## Parser 생성하기
parser = argparse.ArgumentParser()
args = parser.parse_args()

device = torch.device('cuda:'+str(Config.gpu[0]) if torch.cuda.is_available() else 'cpu')
print(device)

## Data Loader
config = Config()
train_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_train.csv', header=None)
val_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_val.csv', header=None)
test_csv = pd.read_csv('/home/connectome/conmaster/Projects/Image_Translation/data_processing/sample_test.csv', header=None)

# sample subjects data: 128 18 36
print(len(train_csv), len(val_csv), len(test_csv))

# split
train_data = DataSplit(data_csv=train_csv, data_dir=config.data_dir, do_transform=True)
val_data = DataSplit(data_csv=val_csv, data_dir=config.data_dir, do_transform=True)
# sub, st, dwi = train_data.__getitem__(3)

# 13184 1854
# print(train_data.__len__(), val_data.__len__())

# load
data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=False, num_workers=16, pin_memory=False)
data_loader_val = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=16, pin_memory=False)

# 412 58
print(len(data_loader_train), len(data_loader_val))

## Model
net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

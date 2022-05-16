

## Parser 생성하기
parser = argparse.ArgumentParser()

parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")
parser.add_argument("--which_epoch", default=0, type=int, help="select which epoch to load")
parser.add_argument("--test_save", default="off", choices=["on","off"],type=str, dest="test_save")

parser.add_argument("--lr", default=2e-4, type=float, dest='lr', help="initial learning rate for adam")
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=200, help='# of iter to linearly decay learning rate to zero')


parser.add_argument("--batch_size", default=16, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
parser.add_argument("--num_workers", default=16, type=int, dest="num_workers")

# Newly added parameters
parser.add_argument("--target", default='mri_info_manufacturersmn', type=str, dest="target")
parser.add_argument("--resize",default=96, required=False,type=int, help='resized to volume having the size of the argument')

parser.add_argument("--ckpt_dir", default='/home/connectome/junb/GANBERT/t1_harmonization/3D-cycleGAN/checkpoint', type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default='/home/connectome/junb/GANBERT/t1_harmonization/3D-cycleGAN/log', type=str, dest="log_dir")
parser.add_argument("--result_dir", default='/home/connectome/junb/GANBERT/t1_harmonization/3D-cycleGAN/result', type=str, dest="result_dir")
parser.add_argument("--gpus", type=int, nargs='+',required=True, dest="gpus_list", help='')

parser.add_argument("--task", default="cyclegan", choices=['cyclegan'], type=str, dest="task")
parser.add_argument("--which_model_netG", default="unet_128", choices=['resnet_9blocks','resnet_6blocks','unet_custom','unet_128','unet_256','Dynet'], type=str, dest="which_model_netG")
parser.add_argument("--which_model_netD", default='basic', choices=['basic','n_layers','pixel'], type=str, dest="which_model_netD")

parser.add_argument('--opts', nargs='+', default=['direction', 0], dest='opts')

parser.add_argument("--nch", default=1, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--wgt_cycle", default=1e1, type=float, dest="wgt_cycle")
parser.add_argument("--wgt_ident", default=5e-1, type=float, dest="wgt_ident")
parser.add_argument("--norm", default='inorm', type=str, dest="norm")

parser.add_argument("--network", default="CycleGAN", choices=['DCGAN', 'pix2pix', 'CycleGAN'], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")
parser.add_argument("--d_loss", default="BCE", choices=["BCE", "wgangp"], type=str, dest="d_loss")

args = parser.parse_args()

device = torch.device('cuda:'+str(Config.gpu[0]) if torch.cuda.is_available() else 'cpu')
print(device)

### Data Loader
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



class Config:
    data_dir = '/scratch/connectome/GANBERT/data/sample_final'
    model_dir = '/home/connectome/conmaster/Pycharm_projects/3D_brain2brain_GAN/cGAN/model'
    log_dir = '/home/connectome/conmaster/Pycharm_projects/3D_brain2brain_GAN/cGAN/log'
    img_dir = '/home/connectome/conmaster/Pycharm_projects/3D_brain2brain_GAN/cGAN/Generated_images'

    gpu = [0]
    mode = 'train'
    restore = None

    nchw = [32,64,64,64]
    G_lr = 2.5e-3
    D_lr = 1e-5
    epoch = 200
    batch_size = 8

    gamma = 0.95
    shuffle = True
    num_workers = 0
    max_iter = 20000
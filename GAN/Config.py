
class Config:
    data_dir = '/scratch/connectome/GANBERT/data/sample/sample_final'
    model_dir = '/home/connectome/conmaster/Pycharm_projects/3D_I2I_GAN/GAN/model'
    log_dir = '/home/connectome/conmaster/Pycharm_projects/3D_I2I_GAN/GAN/log'
    img_dir = '/home/connectome/conmaster/Pycharm_projects/3D_I2I_GAN/GAN/Generated_images'

    gpu = [2]
    mode = 'train'
    restore = None

    nchw = [32,64,64,64]
    G_lr = 2.5e-3
    D_lr = 1e-5
    epoch = 200
    batch_size = 32

    gamma = 0.95
    shuffle = False
    num_workers = 0
    max_iter = 200
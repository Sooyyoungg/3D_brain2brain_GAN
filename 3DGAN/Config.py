
class Config:
    data_dir = '/scratch/connectome/GANBERT/data/sample/final'
    model_dir = '/home/connectome/conmaster/Pycharm_projects/GAN_study/3DGAN/model'
    log_dir = '/home/connectome/conmaster/Pycharm_projects/GAN_study/3DGAN/log'
    img_dir = '/home/connectome/conmaster/Pycharm_projects/GAN_study/3DGAN/image'

    nchw = [32,64,64,64]
    G_lr = 2.5e-3
    D_lr = 1e-5
    step_size = 2000
    gamma = 0.95
    shuffle = True
    num_workers = 0
    max_iter = 20000
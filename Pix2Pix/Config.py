class Config:
    ## dataset parameters
    train_list = '/scratch/connectome/conmaster/Projects/Image_Translation/final_preprocessing/sample_train_subjects.csv'
    valid_list = '/scratch/connectome/conmaster/Projects/Image_Translation/final_preprocessing/sample_val_subjects.csv'
    test_list = '/scratch/connectome/conmaster/Projects/Image_Translation/final_preprocessing/sample_test_subjects.csv'
    t1_dir = '/storage/connectome/GANBERT/data/T1'
    b0_dir = '/storage/connectome/GANBERT/data/B0'
    dwi_dir = '/storage/connectome/GANBERT/data/DWI'
    grad_dir = '/storage/connectome/GANBERT/data/Gradient'

    # output directory
    log_dir = '/scratch/connectome/conmaster/Pycharm_projects/3D_brain2brain_GAN/Pix2Pix/log'
    img_dir = '/scratch/connectome/conmaster/Pycharm_projects/3D_brain2brain_GAN/Pix2Pix/Generated_images'
    valid_img_dir = '/scratch/connectome/conmaster/Pycharm_projects/3D_brain2brain_GAN/Pix2Pix/Best_Train_images'
    test_img_dir = '/scratch/connectome/conmaster/Pycharm_projects/3D_brain2brain_GAN/Pix2Pix/Tested_images'

    ## basic parameters
    gpu_ids = [7]
    n_epoch = 100
    n_iter = 100
    n_iter_decay = 100
    batch_size = 64
    lr = 0.0002
    lr_policy = 'step'
    lr_decay_iters = 50
    beta1 = 0.5
    pool_size = 50
    image_display_iter = 100
    gan_mode = 'vanilla'
    lambda_L1 = 100

    # model parameters
    input_nc = 2
    output_nc = 1
    ngf = 64
    ndf = 64
    initial = True        # Initialize the Generator
    norm = 'instance'     # [instance | batch | none]
    init_type = 'normal'  # [normal | xavier | kaiming | orthogonal]
    init_gain = 0.02      # scaling factor for normal, xavier and orthogonal
    no_dropout = 'store_true'   # no dropout for generator
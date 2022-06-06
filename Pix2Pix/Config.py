
class Config:
    ## dataset parameters
    data_dir = '/storage/connectome/GANBERT/data/sample/sample_b0_input_ver' # input으로 사용할 image 경로
    img_dir = '/scratch/connectome/conmaster/Pycharm_projects/3D_brain2brain_GAN/Pix2Pix/Generated_images'  # 저장할 image의 경로

    ## basic parameters
    n_epoch = 100
    batch_size = 32
    lr = 0.00005
    gpu = [0]
    image_display_iter=100

    ## model parameters
    input_nc = 3        # # of input channel
    output_nc = 3       # # of output channel
    ngf = 64            # # of generator filters in last conv layer
    ndf = 64            # # of discriminator filters in the first conv layer
    netG = 'resnet_9blocks'
    netD = 'basic'
    n_layers_D = 3      # only used if netD==n_layers
    norm = 'instance'   # [instance | batch | none]
    init_type = 'normal' # [normal | xavier | kaiming | orthogonal]
    init_gain = 0.02    # scaling factor for normal, xavier and orthogonal
    no_dropout = 'store_true'   # no dropout for generator



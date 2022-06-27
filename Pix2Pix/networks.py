import functools
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

###############################################################################
# Helper Functions
###############################################################################
def get_scheduler(optimizer, config):
    if config.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + config.n_epoch - config.n_iter) / float(config.n_iter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iters, gamma=0.1)
    elif config.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_iter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config.lr_policy)
    return scheduler

def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    # init_type: normal, xavier, kaiming
    def init_func(m):
        classname = m.__class__.__name__
        # Initialize weights of Convolution layer or Linear layer
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fain_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            # bias
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # Initialize weights of Batch normalization layer
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], initial=True):
    if torch.cuda.is_available():
        if len(gpu_ids) > 0:
            # assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)
    else:
        net.to('cpu')
    if initial:
        init_weights(net, init_type, init_gain=init_gain)
    return net

def define_G(initial=True, init_type='normal', init_gain=0.02, gpu_ids=[]):
    try:
        net = UnetGenerator()
    except:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids, initial)

def define_D(input_nc, ndf, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    try:
        net = NLayerDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    except:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

###############################################################################
# Generator & Discriminator & Mapping Networks
###############################################################################

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class MappingNetwork(nn.Module):
    def __init__(self):
        super(MappingNetwork, self).__init__()

        layers = []
        layers += [nn.Linear(1, 64), nn.ReLU()]
        layers += [nn.Linear(62, 128), nn.ReLU()]
        layers += [nn.Linear(128, 256), nn.ReLU()]
        layers += [nn.Linear(256, 1)]

        self.mapping = nn.Sequential(*layers)

    def forward(self, cond):
        return self.mapping(cond)

class UnetGenerator(nn.Module):
    def __init__(self):
        super(UnetGenerator, self).__init__()

        # Conv - BatchNorm - ReLU
        def CBR2d(in_channels=2, out_channels=64, kernel=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)
            return cbr

        ## Contracting Path
        self.enc1_1 = CBR2d(in_channels=2, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)

        ## Expansing path
        self.dec3_1_C = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec3_1_A = AdaIN(style_dim=1, num_features=128 + 1)
        self.dec3_1_R = nn.ReLU()

        self.unpool2 = nn.ConvTranspose2d(in_channels=128 + 1, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2_C = nn.Conv2d(in_channels=2 * 128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec2_2_A = AdaIN(style_dim=1, num_features=128 + 1)
        self.dec2_2_R = nn.ReLU()

        self.dec2_1_C = nn.Conv2d(in_channels=128 + 1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec2_1_A = AdaIN(style_dim=1, num_features=64 + 1)
        self.dec2_1_R = nn.ReLU()

        self.unpool1 = nn.ConvTranspose2d(in_channels=64 + 1, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2_C = nn.Conv2d(in_channels=2 * 64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec1_2_A = AdaIN(style_dim=1, num_features=64 + 1)
        self.dec1_2_R = nn.ReLU()

        self.dec1_1 = CBR2d(in_channels=64 + 1, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.tanh = nn.Tanh()


    def forward(self, input, cond):
        cond = MappingNetwork(cond)

        ## Encoder
        enc1_1 = self.enc1_1(input)
        enc1_2 = self.enc1_2(enc1_1)
        pool = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)

        ## Decoder
        dec3_1_C = self.dec3_1_C(enc3_1)
        dec3_1_A = self.dec3_1_A(dec3_1_C, cond)
        dec3_1_R = self.dec3_1_R(dec3_1_A)

        unpool2 = self.unpool2(dec3_1_R)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2_C = self.dec2_2_C(cat2)
        dec2_2_A = self.dec2_2_A(dec2_2_C, cond)
        dec2_2_R = self.dec2_2_R(dec2_2_A)

        dec2_1_C = self.dec2_1_C(dec2_2_R)
        dec2_1_A = self.dec2_1_A(dec2_1_C, cond)
        dec2_1_R = self.dec2_1_R(dec2_1_A)

        unpool1 = self.unpool1(dec2_1_R)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2_C = self.dec1_2_C(cat1)
        dec1_2_A = self.dec1_2_A(dec1_2_C, cond)
        dec1_2_R = self.dec1_2_R(dec1_2_A)

        dec1_1 = self.dec1_1(dec1_2_R)

        fc = self.fc(dec1_1)
        output = self.tanh(fc)

        return output


## Discriminator
# Defines a PatchGAN discriminator
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Concatenator
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, True)]
        sequence += [
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * 4, ndf * 4, kernel_size=6, stride=1, padding=1)]
        sequence += [
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


##############################################################################
# Loss function
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
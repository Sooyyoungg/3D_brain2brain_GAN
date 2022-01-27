import torch
import torch.nn as nn
from Blocks import  Conv3dBlock, ResBlocks

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Encoder = ResEncoder()
        self.Decoder = Decoder()
        self.generate = nn.ModuleList()
        # struct: (1, 190, 190, 190) -> (256, 16, 16, 16)
        self.generate.append(self.Encoder)
        # (256, 16, 16, 16) -> (1, 190, 190, 190)
        self.generate.append(self.Decoder)

    def forward(self, struct_image, gradient):
        # input structure image: (batch_size, 1, 190, 190, 190)
        gen_latent = self.generate[0](struct_image)
        self.gen_dwi = self.generate[1](gen_latent, gradient)
        """for code in self.generate:
            self.gen_dwi = code(struct_image, gradient)"""
        # output: (batch_size, 1, 190, 190, 190)
        return self.gen_dwi

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.Discriminate = nn.Sequential(
            # (1, 190, 190, 190) -> (32, 95, 95, 95)
            nn.Conv3d(1, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),

            # (32, 95, 95, 95) -> (64, 47, 47, 47)
            nn.Conv3d(32, 64, 5, 2, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),

            # (64, 47, 47, 47) -> (128, 23, 23, 23)
            nn.Conv3d(64, 128, 5, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),

            # (128, 23, 23, 23) -> (256, 11, 11, 11)
            nn.Conv3d(128, 256, 5, 2, 1),
            nn.BatchNorm3d(236),
            nn.LeakyReLU(0.2),

            # (256, 11, 11, 11) -> (512, 5, 5, 5)
            nn.Conv3d(256, 512, 5, 2, 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
        )
        self.LinSigmoid = nn.Sequential(
            # (512, 5, 5, 5) -> (1, 1, 1, 1)
            # nn.Conv3d(512, 1, 5, 2, 0),
            nn.Linear(512 * 5 * 5 * 5, 1),
            nn.Sigmoid()
        )

    def forward(self, dwi):
        # input: (batch_size, 1, 190, 190, 190)
        result = self.Discriminate(dwi)
        result = result.view(-1, 512 * 5 * 5 * 5)
        result = self.LinSigmoid(result)

        print("discriminate result shape:", result.shape)
        print("result:", result)
        return result


####################### Encoder & Decoder #######################
class ResEncoder(nn.Module):
    def __init__(self, norm='none', activ='relu', pad_type='zero'):
        super(ResEncoder, self).__init__()
        self.input_dim = 1
        self.dim = 8
        self.n_downsample = 5
        self.n_res = 256

        self.model = []
        # (1, 128, 128, 128) -> (8, 128, 128, 128)
        self.model += [Conv3dBlock(self.input_dim, self.dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks : image size 절반으로 줄어듦
        # (8, 128, 128, 128) -> (16, 64, 64, 64) -> (32, 32, 32, 32) -> (64, 16, 16, 16) -> (128, 8, 8, 8) -> (256, 4, 4, 4)
        for i in range(self.n_downsample):
            self.model += [Conv3dBlock(self.dim, 2 * self.dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            self.dim *= 2
        # residual blocks
        # (256, 4, 4, 4) -> (256, 4, 4, 4)
        self.model += [ResBlocks(self.n_res, self.dim, norm=norm, activation=activ, pad_type=pad_type)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = self.dim

    def forward(self, x):
        print("output channel: ", self.output_dim)
        print("Encoder output dim: ", x.shape)
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, res_norm='none', activ='tanh', pad_type='zero'):
        super(Decoder, self).__init__()

        ### Gradient mapping space
        self.grad_input_dim = 4
        self.grad_output_dim = 4 * 4 * 4
        self.grad_fc = 4

        self.grad_mapping = []
        for i in range(self.grad_fc):
            self.grad_mapping += [nn.Linear(self.grad_input_dim, self.grad_input_dim * 2)]
            self.grad_input_dim *= 2
        self.grad_mapping = nn.Sequential(*self.grad_mapping)

        ### Generate fake image
        self.output_dim = 1
        self.dim = 256
        self.n_upsample = 4
        self.n_res = 256

        self.model = []
        # (257, 4, 4, 4) -> (256, 4, 4, 4)
        self.model += [ResBlocks(self.n_res, self.dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        # (256, 4, 4, 4) -> (128, 8, 8, 8) -> (64, 16, 16, 16) -> (32, 32, 32, 32) -> (16, 64, 64, 64)
        for i in range(self.n_upsample):
            self.model += [nn.Upsample(scale_factor=2, mode='nearest'),
                           Conv3dBlock(self.dim, self.dim // 2, 5, 1, 2, norm='ln', activation='relu', pad_type=pad_type)]
            self.dim //= 2
        # (16, 64, 64, 64) -> (16, 64, 64, 64)
        self.model += [Conv3dBlock(self.dim, self.dim, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        # (16, 64, 64, 64) -> (16, 128, 128, 128)
        self.model += [nn.Upsample(scale_factor=2, mode='nearest'),
                       Conv3dBlock(self.dim, self.dim, 5, 1, 2, norm='ln', activation='relu', pad_type=pad_type)]
        # use reflection padding in the last conv layer
        # (16, 128, 128, 128) -> (1, 128, 128, 128)
        self.model += [Conv3dBlock(self.dim, self.output_dim, 7, 1, 3, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [nn.Tanh()]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, gradient):
        gradient = self.grad_mapping(gradient)
        x = torch.cat([x, gradient], dim=0)
        print("Decoder output dim: ", x.shape)
        return self.model(x)
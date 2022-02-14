import torch
import torch.nn as nn
from Blocks import Conv3dBlock, ResBlocks
from Config import Config

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Encoder = ResEncoder()
        self.Decoder = Decoder()
        self.generate = nn.ModuleList()
        # struct: (1, 64, 64, 64) -> (128, 4, 4, 4)
        self.generate.append(self.Encoder)
        # (128, 4, 4, 4) -> (1, 64, 64, 64)
        self.generate.append(self.Decoder)

    def forward(self, struct_image, gradient):
        # input structure image: (batch_size, 1, 64, 64, 64)
        gen_latent = self.generate[0](struct_image)
        self.gen_dwi = self.generate[1](gen_latent, gradient)
        # output: (batch_size, 1, 64, 64, 64)
        return self.gen_dwi

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.Discriminate = nn.Sequential(
            # (1, 64, 64, 64) -> (32, 32, 32, 32)
            nn.Conv3d(1, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),

            # (32, 32, 32, 32) -> (64, 16, 16, 16)
            nn.Conv3d(32, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),

            # (64, 16, 16, 16) -> (128, 8, 8, 8)
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),

            # (128, 8, 8, 8) -> (256, 4, 4, 4)
            nn.Conv3d(128, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
        )
        self.LinSigmoid = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, dwi):
        # input: torch.Size([batch_size, 1, 64, 64, 64])
        result = self.Discriminate(dwi)
        result = result.view(-1, 256 * 4 * 4 * 4)
        result = self.LinSigmoid(result)

        #print("discriminate result shape:", result.shape)  # torch.Size([16, 1])
        #print("result:", result)
        return result


####################### Encoder & Decoder #######################
class ResEncoder(nn.Module):
    def __init__(self, norm='none', activ='relu', pad_type='zero'):
        super(ResEncoder, self).__init__()
        self.input_dim = 1
        self.dim = 8
        self.n_downsample = 4
        self.n_res = 2

        self.model = []
        # (1, 64, 64, 64) -> (8, 64, 64, 64)
        self.model += [Conv3dBlock(self.input_dim, self.dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks : image size 절반으로 줄어듦
        # (8, 64, 64, 64) -> (16, 32, 32, 32) -> (32, 16, 16, 16) -> (64, 8, 8, 8) -> (128, 4, 4, 4)
        for i in range(self.n_downsample):
            self.model += [Conv3dBlock(self.dim, 2 * self.dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            self.dim *= 2
        # residual blocks
        # (128, 4, 4, 4) -> (128, 4, 4, 4)
        self.model += [ResBlocks(self.n_res, self.dim, norm=norm, activation=activ, pad_type=pad_type)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = self.dim

    def forward(self, x):
        output = self.model(x)
        #print("Encoder output dim: ", output.shape)  # torch.Size([32, 128, 4, 4, 4])
        return output

class Decoder(nn.Module):
    def __init__(self, res_norm='none', activ='tanh', pad_type='zero'):
        super(Decoder, self).__init__()

        ### Gradient mapping space
        self.grad_input_dim = 4
        self.grad_output_dim = 1 * 4 * 4 * 4
        self.grad_fc = 4

        self.grad_mapping = []
        for i in range(self.grad_fc):
            self.grad_mapping += [nn.Linear(self.grad_input_dim, self.grad_input_dim * 2)]
            self.grad_input_dim *= 2
        self.grad_mapping = nn.Sequential(*self.grad_mapping)

        ### Generate fake image
        self.output_dim = 1
        self.dim = 129
        self.n_upsample = 4
        self.n_res = 2

        self.model = []
        # (129, 4, 4, 4) -> (129, 4, 4, 4)
        self.model += [ResBlocks(self.n_res, self.dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        # (129, 4, 4, 4) -> (64, 8, 8, 8) -> (32, 16, 16, 16) -> (16, 32, 32, 32) -> (8, 64, 64, 64)
        for i in range(self.n_upsample):
            self.model += [nn.Upsample(scale_factor=2, mode='nearest'),
                           Conv3dBlock(self.dim, self.dim // 2, 5, 1, 2, norm='ln', activation='lrelu', pad_type=pad_type)]
            self.dim //= 2
        # (8, 64, 64, 64) -> (8, 64, 64, 64)
        self.model += [Conv3dBlock(self.dim, self.dim, 5, 1, 2, norm='ln', activation='lrelu', pad_type=pad_type)]
        # use reflection padding in the last conv layer
        # (8, 64, 64, 64) -> (1, 64, 64, 64)
        self.model += [Conv3dBlock(self.dim, self.output_dim, 7, 1, 3, norm='none', activation='lrelu', pad_type=pad_type)]
        self.model += [nn.Sigmoid()]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, gradient):
        batch_size = Config.batch_size
        gradient = self.grad_mapping(gradient)
        gradient = torch.reshape(gradient, [batch_size, 1, 4, 4, 4])
        x = torch.cat([x, gradient], dim=1)
        output = self.model(x)
        #print("Decoder output dim: ", output.shape)  #torch.Size([32, 128, 4, 4, 4])
        return output
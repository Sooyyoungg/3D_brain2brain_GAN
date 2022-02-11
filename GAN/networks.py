import torch
import torch.nn as nn
from numpy import inf
from Blocks import  Conv3dBlock, ResBlocks

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

    def forward(self, struct_image):
        # input structure image: (batch_size, 1, 64, 64, 64)
        gen_latent = self.generate[0](struct_image)
        self.gen_dwi = self.generate[1](gen_latent)
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
        #print("Discriminator input shape: ", dwi.shape)
        result = self.Discriminate(dwi)
        result = result.view(-1, 256 * 4 * 4 * 4)
        result = self.LinSigmoid(result)

        # output: torch.Size([batch_size, 1])
        #print("Discriminator result shape:", result.shape)
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
        #print("output channel: ", self.output_dim)
        # Encoder output: torch.Size([batch_size, 1, 64, 64, 64])
        #print("Generator - Encoder output dim: ", x.shape)
        #print(torch.isnan(x).any())
        output = self.model(x)
        #print(torch.isnan(output).any()
        """if inf in output or -inf in output:
            out_copy = output.clone().detach()
            out_copy[output == -inf] = inf
            output[output == -inf] = torch.min(out_copy)
            out_copy[output == inf] = -inf
            output[output == inf] = torch.max(out_copy)"""
        if inf in output or -inf in output:
            print("inf yes", torch.min(output), torch.max(output))
        if torch.isnan(output).any():
            print("nan yes")
        return output

class Decoder(nn.Module):
    def __init__(self, res_norm='none', activ='tanh', pad_type='zero'):
        super(Decoder, self).__init__()

        ### Generate fake image
        self.output_dim = 1
        self.dim = 128
        self.n_upsample = 4
        self.n_res = 2

        self.model = []
        # (128, 4, 4, 4) -> (128, 4, 4, 4)
        self.model += [ResBlocks(self.n_res, self.dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        # (128, 4, 4, 4) -> (64, 8, 8, 8) -> (32, 16, 16, 16) -> (16, 32, 32, 32) -> (8, 64, 64, 64)
        for i in range(self.n_upsample):
            self.model += [nn.Upsample(scale_factor=2, mode='nearest'),
                           Conv3dBlock(self.dim, self.dim // 2, 5, 1, 2, norm='ln', activation='relu', pad_type=pad_type)]
            self.dim //= 2
        # (8, 64, 64, 64) -> (8, 64, 64, 64)
        self.model += [Conv3dBlock(self.dim, self.dim, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        # use reflection padding in the last conv layer
        # (8, 64, 64, 64) -> (1, 64, 64, 64)
        self.model += [Conv3dBlock(self.dim, self.output_dim, 7, 1, 3, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [nn.Sigmoid()]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        # Decoder output: torch.Size([batch_size, 128, 64, 64, 64])
        #print("Generator - Decoder output dim: ", x.shape)
        output = self.model(x)
        #print("Decoder: ", torch.isnan(x).any(), torch.isnan(output).any())
        if inf in output or -inf in output:
            print("inf yes", torch.min(output), torch.max(output))
        if torch.isnan(output).any():
            print("nan yes")
        return output
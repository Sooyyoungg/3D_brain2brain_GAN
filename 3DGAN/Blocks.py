from torch import nn

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv3dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv3dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out = out + residual
        return out

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()

        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class Conv3dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride, padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv3dBlock, self).__init__()
        self.use_bias = True

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad3d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad3d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad3d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm3d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm3d(norm_dim)
        #elif norm == 'ln':
        #    self.norm = LayerNorm(norm_dim)
        #elif norm == 'adain':
        #    self.norm = AdaptiveInstanceNorm3d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        """if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv3d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            """
        self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
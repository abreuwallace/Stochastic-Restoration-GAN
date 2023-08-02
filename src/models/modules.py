import torch
import torch.nn as nn

class SelfGating(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.C = channels
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.prelu(x[:, :self.C//2, :, :]) * self.sigmoid(x[:, self.C//2:, :, :])
        return x

class EncodingBlock(nn.Module):
    def __init__(self, channels, k_sizes, dilation, padding):
        """ Generator encoding section plus reshaping and remapping.
        Args:
            channels (list[int]): list of channel sizes from first input to last output 
            k_sizes (list[int]): list of layers' kernel sizes
            dilation (list[int]): list of layers' dilation sizes
            padding (list[int]): list of layers' padding sizes
        """ 
        super().__init__()
        #Hardcode for testing (generator)
        #channels = [2, 18, 38, 38, 4096, 128, 256]
        #k_sizes = [3, 3, 3, (1024, 1), 1]
        #dilation = [1, 2, 4, 1, 1]
        #padding = [1, 2, 4, 0, 0]

        #number of layers excluding reshaping and remapping
        n_enc_layers = len(k_sizes) - 1
        encoder_layers = []
        for i in range(n_enc_layers):
            conv_layer = [nn.Conv2d(channels[i], channels[i+1], kernel_size=k_sizes[i], dilation=dilation[i], padding=padding[i]),
                          nn.PReLU()]
            encoder_layers += conv_layer

        self.enc_block = nn.Sequential(*encoder_layers)
        self.remap = nn.Sequential(nn.Conv2d(channels[-2], channels[-1], kernel_size=k_sizes[-1], dilation=dilation[-1], padding=padding[-1]),
                                   nn.PReLU())

    def forward(self, x):
        x = self.enc_block(x)
        B, _, _, T = x.shape
        x = torch.reshape(x, (B, 128, 32, T))
        x = self.remap(x)
        return x
    
class IntermediateBlock(nn.Module):
    def __init__(self, channels, k_sizes, dilation, padding):
        """ Generator intermediate section.
        Args:
            channels (list[int]): list of channel sizes from first input to last output 
            k_sizes (list[int]): list of layers' kernel sizes
            dilation (list[int]): list of layers' dilation sizes
            padding (list[int]): list of layers' padding sizes
        """ 
        super().__init__()
        #Hardcode for testing (generator)
        #channels = [256, 320, 256, 128, 256, 128, 256, 128, 256, 128, 256,
        #            128, 256, 128, 256, 128, 256, 128, 256]
        #k_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        #dilation = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
        #padding = [(1, 0), (2, 0), (4, 0), (8, 0), (16, 0),
        #           (1, 0), (2, 0), (4, 0), (8, 0), (16, 0)]

        self.pre_noiseconv = nn.Sequential(nn.Conv2d(channels[0], channels[0], kernel_size=k_sizes[0], dilation=dilation[0], padding=padding[0]),
                                   nn.PReLU())
        self.gating = SelfGating(channels[0])
        n_selfgating_layers = len(k_sizes) - 1
        self.conv_layers = nn.ModuleList()
        for i in range(1, n_selfgating_layers + 1):
            conv_selfgating = nn.Sequential(nn.Conv2d(channels[2*i-1], channels[2*i], kernel_size=k_sizes[i], dilation=dilation[i], padding=padding[i]))
            self.conv_layers.insert(i, conv_selfgating)
        
                
    def forward(self, x, z):
        x = self.pre_noiseconv(x)
        B, _, F, T = x.shape
        z = z.expand(B, 64, F, T)
        x = torch.cat((x, z), dim=1)
        for layer in self.conv_layers:
            x = layer(x)
            x = self.gating(x)
        return x

class DecodingBlock(nn.Module):
    def __init__(self, channels, k_sizes, dilation, padding):
        """ Generator decoding section.
        Args:
            channels (list[int]): list of channel sizes from first input to last output 
            k_sizes (list[int]): list of layers' kernel sizes
            dilation (list[int]): list of layers' dilation sizes
            padding (list[int]): list of layers' padding sizes
        """ 
        super().__init__()
        #Hardcode for testing (generator)
        #channels = [4096, 38, 38, 18, 2]
        #k_sizes = [(1024, 1), 3, 3, 3]
        #dilation = [1, 4, 2, 1]
        #padding = [0, 4, 2, 1]
        decoder_layers = []
        n_dec_layers = len(k_sizes)
        for i in range(n_dec_layers):
            deconv_layer = [nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=k_sizes[i], dilation=dilation[i], padding=padding[i]),
                          nn.PReLU()]
            decoder_layers += deconv_layer
        self.deconv_block = nn.Sequential(*decoder_layers)
    def forward(self, x):
        B, C, F, T = x.shape
        x = torch.reshape(x, (B, C * F, 1, T))
        x = self.deconv_block(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, channels, k_sizes, dilation, padding):
        """ Discriminator intermediate with skip connections section.
        Args:
            channels (list[int]): list of channel sizes from first input to last output 
            k_sizes (list[int]): list of layers' kernel sizes
            dilation (list[int]): list of layers' dilation sizes
            padding (list[int]): list of layers' padding sizes
        """ 
        super().__init__()
        #Hardcode for testing 
        #channels = [256, 256, 256, 128, 256, 128, 256, 128, 256, 128, 256,
        #            128, 256, 128, 256, 128, 256, 128, 256]
        #k_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        #dilation = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
        #padding = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16),
        #           (1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]

        self.pre_noiseconv = nn.Sequential(nn.Conv2d(channels[0], channels[0], kernel_size=k_sizes[0], dilation=dilation[0], padding=padding[0]),
                                   nn.PReLU())
        self.gating = SelfGating(channels[0])
        n_selfgating_layers = 9
        self.conv_layers = nn.ModuleList()
        for i in range(1, n_selfgating_layers + 1):
            conv_selfgating = nn.Sequential(nn.Conv2d(channels[2*i-1], channels[2*i], kernel_size=k_sizes[i], dilation=dilation[i], padding=padding[i]))
            self.conv_layers.insert(i, conv_selfgating)
        
                
    def forward(self, x):
        x = self.pre_noiseconv(x)
        x_res = 0
        for layer in self.conv_layers:
            x_skip = x_res
            x_res = layer(x) + x_skip
            x = self.gating(x_res)
        return x
    
class LogitBlock(nn.Module):
    def __init__(self, channels, k_sizes, dilation, padding):
        """ Discriminator output section.
        Args:
            channels (list[int]): list of channel sizes from first input to last output 
            k_sizes (list[int]): list of layers' kernel sizes
            dilation (list[int]): list of layers' dilation sizes
            padding (list[int]): list of layers' padding sizes
        """ 
        super().__init__()
        #Hardcode for testing 
        #channels = [128, 256, 1]
        #k_sizes = [3, (32, 1)]
        #dilation = [1, 1]
        #padding = [1, 0]

        output_layers = []
        for i in range(len(channels) - 1):
            out_layer = [nn.Conv2d(channels[i], channels[i+1], kernel_size=k_sizes[i], dilation=dilation[i], padding=padding[i]),
                          nn.PReLU()]
            output_layers += out_layer

        self.logit_block = nn.Sequential(*output_layers)

    def forward(self, x):
        x = self.logit_block(x)
        return x
import torch
from torch import nn
from src.models.modules import EncodingBlock, ResidualBlock, LogitBlock

class Discriminator(nn.Module):
    def __init__(self, enc_params, res_params, logit_params):
        """ Discriminator model.
        Args:
            enc_params (dict): dictionary of parameters for the encoding section
            res_params (dict): dictionary of parameters for the residual section
            logit_params (dict): dictionary of parameters for the output section
        """ 
        super().__init__()
        #Hardcode for testing
        # enc_params = dict(channels = [2, 18, 38, 38, 4096, 128, 256],
        #                   k_size = [3, 3, 3, (1024, 1), 1],
        #                   dilation = [3, 3, 3, (1024, 1), 1],
        #                   padding = [1, 2, 4, 0, 0])

        # res_params = dict(channels = [256, 256, 256, 128, 256, 128, 256, 128, 256, 128, 256,
        #                               128, 256, 128, 256, 128, 256, 128, 256]
        #                   k_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        #                   dilation = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
        #                   padding = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16),
        #                             (1, 1), (2, 2), (4, 4), (8, 8), (16, 16)])

        # logit_params = dict(channels=[128, 256, 1],
        #                     k_sizes=[3, (32, 1)],
        #                     dilation=[1, 1],
        #                     padding=[1, 0])

        self.encoding_block = EncodingBlock(enc_params["channels"], 
                                            enc_params["k_sizes"], 
                                            enc_params["dilation"], 
                                            enc_params["padding"])
        
        self.residual_block = ResidualBlock(res_params["channels"], 
                                            res_params["k_sizes"], 
                                            res_params["dilation"], 
                                            res_params["padding"])
        
        self.logit_block = LogitBlock(logit_params["channels"], 
                                      logit_params["k_sizes"], 
                                      logit_params["dilation"], 
                                      logit_params["padding"])
        

    def forward(self, x):
        x = self.encoding_block(x)
        x = self.residual_block(x)
        x = self.logit_block(x)
        return x

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean(dr)
        g_loss = torch.mean(dg)
        loss += (r_loss - g_loss)

    return loss  

def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        g_loss = torch.mean(dg)
        loss += g_loss
    return loss
import torch
from torch import nn
from src.models.modules import EncodingBlock, IntermediateBlock, DecodingBlock

class Generator(nn.Module):
    def __init__(self, enc_params, interm_params, dec_params):
        """ Generator model.
        Args:
            enc_params (dict): dictionary of parameters for the encoding section
            inter_params (dict): dictionary of parameters for the intermediate section
            dec_params (dict): dictionary of parameters for the decoding section
        """ 
        super().__init__()
        #Hardcode for testing
        # enc_params = dict(channels = [2, 18, 38, 38, 4096, 128, 256],
        #                   k_size = [3, 3, 3, (1024, 1), 1],
        #                   dilation = [1, 2, 4, 1, 1],
        #                   padding = [1, 2, 4, 0, 0])

        # int_params = dict(channels = [256, 320, 256, 128, 256, 128, 256, 128, 256, 128, 256,
        #                               128, 256, 128, 256, 128, 256, 128, 256]
        #                   k_size = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #                   dilation = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16],
        #                   padding = [(1, 0), (2, 0), (4, 0), (8, 0), (16, 0),
        #                              (1, 0), (2, 0), (4, 0), (8, 0), (16, 0)])

        # dec_params = dict(channels = [4096, 38, 38, 18, 2],
        #                   k_size = [(1024, 1), 3, 3, 3],
        #                   dilation = [1, 4, 2, 1],
        #                   padding = [0, 4, 2, 1])

        self.encoding_block = EncodingBlock(enc_params["channels"], 
                                            enc_params["k_sizes"], 
                                            enc_params["dilation"], 
                                            enc_params["padding"])
        
        self.intermediate_block = IntermediateBlock(interm_params["channels"], 
                                                    interm_params["k_sizes"], 
                                                    interm_params["dilation"], 
                                                    interm_params["padding"])
        
        self.decoding_block = DecodingBlock(dec_params["channels"], 
                                            dec_params["k_sizes"], 
                                            dec_params["dilation"], 
                                            dec_params["padding"])

    def forward(self, x, z):
        x = self.encoding_block(x)
        x = self.intermediate_block(x, z)
        x = self.decoding_block(x)
        return x

"""
Example UNet using MinkowskiEngine
"""
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from collections import OrderedDict
import logging
import sys

# set up logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("protodune_cosmic.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

activations = {
    'relu':     ME.MinkowskiReLU(),
    'tanh':     ME.MinkowskiTanh(),
    'sigmoid':  ME.MinkowskiSigmoid(),
    'softmax':  ME.MinkowskiSoftmax(),
}

def get_activation(
    activation: str,
):
    if activation in activations.keys():
        return activations[activation]

class DoubleConv2d(ME.MinkowskiNetwork):
    """
    """
    def __init__(self,
        name, 
        in_channels, 
        out_channels,
        kernel_size:    int=3,
        stride:         int=1,
        dilation:       int=1,
        activation:     str='relu',
        batch_norm:     bool=True,
        dimension:      int=3, 
    ):
        """
        """
        super(DoubleConv2d, self).__init__(dimension)
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dimension = dimension
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bias = False
        else:
            self.bias = True
        self.activation = activation
        self.activation_fn = get_activation(self.activation)
        logging.info(f"Creating DoubleConv2dLayer {self.name} with in_channels: {self.in_channels}, out_channels: {self.out_channels} and dimension: {self.dimension}.")
        logging.info(f"DoubleConv2dLayer {self.name} has activation function: {self.activation}, bias: {self.bias} and batch_norm: {self.batch_norm}.")
        logging.info(f"DoubleConv2dLayer {self.name} has kernel_size: {self.kernel_size}, stride: {self.stride} and dilation: {self.dilation}.")
        self.construct_model()

    def construct_model(self):
        """
        Create model dictionary
        """
        _dict = OrderedDict()
        # create conv layer
        _dict[f'{self.name}_conv1'] = ME.MinkowskiConvolution(
            in_channels  = self.in_channels,
            out_channels = self.out_channels,
            kernel_size  = self.kernel_size,
            stride       = self.stride,
            dilation     = self.dilation,
            bias         = self.bias,
            dimension    = self.dimension
        )
        if self.batch_norm:
            _dict[f'{self.name}_batch_norm1'] = ME.MinkowskiBatchNorm(self.out_channels)
        _dict[f'{self.name}_{self.activation}1'] = self.activation_fn
        # second conv layer
        _dict[f'{self.name}_conv2'] = ME.MinkowskiConvolution(
            in_channels  = self.out_channels,
            out_channels = self.out_channels,
            kernel_size  = self.kernel_size,
            stride       = self.stride,
            dilation     = self.dilation,
            bias         = self.bias,
            dimension    = self.dimension
        )
        if self.batch_norm:
            _dict[f'{self.name}_batch_norm2'] = ME.MinkowskiBatchNorm(self.out_channels)
        _dict[f'{self.name}_{self.activation}2'] = self.activation_fn
        self.module_dict = nn.ModuleDict(_dict)
        logging.info(f"Constructed DoubleConv2dLayer: {self.module_dict}.")

    def forward(self, 
        x
    ):
        """
        Iterate over the module dictionary.
        """
        for layer in self.module_dict.keys():
            x = self.module_dict[layer](x)
        return x

if __name__ == "__main__":

    doubleconv = DoubleConv2d(
        name='test_layer',
        in_channels=1,
        out_channels=1,
        dimension=3
    )
    coords = []
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if np.random.uniform(0,1,1)[0] > 0.5:
                    coords.append([i,j,k])
    coords = [np.array(coords)]
    coords = ME.utils.batched_coordinates(coords)

    N = len(coords)
    dtype=torch.float32
    feats = torch.arange(N * 1).view(N, 1).to(dtype)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    input = ME.SparseTensor(feats, coords)

    y = doubleconv(input)

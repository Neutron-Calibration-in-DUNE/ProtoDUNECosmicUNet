"""
Example UNet using MinkowskiEngine
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
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

    # Here are a set of standard UNet parameters, which must be 
# adjusted by the user for each application
UNet_params = {
    'in_channels':  1,
    'out_channels': 1,  # this is the number of classes for the SS
    'filtrations':  [64, 128, 256, 512],    # the number of filters in each downsample
    # standard double_conv parameters
    'double_conv_kernel':   3,
    'double_conv_stride':   1,
    'double_conv_dilation': 1,
    'double_conv_activation':   'relu',
    'double_conv_dimension':    3,
    'double_conv_batch_norm':   True,
    # conv transpose layers
    'conv_transpose_kernel':    2,
    'conv_transpose_stride':    2,
    'conv_transpose_dilation':  1,
    'conv_transpose_dimension': 3,
    # max pooling layer
    'max_pooling_kernel':   2,
    'max_pooling_stride':   2,
    'max_pooling_dilation': 1,
    'max_pooling_dimension':3,
}

class UNet(nn.Module):
    """
    """
    def __init__(self,
        cfg:    dict=UNet_params   # configuration parameters
    ):
        super(UNet, self).__init__()
        self.cfg = cfg
        # check cfg
        logging.info(f"checking UNet architecture using cfg: {self.cfg}")
        for item in UNet_params.keys():
            if item not in self.cfg:
                logging.error(f"parameter {item} was not specified in config file {self.cfg}")
                raise AttributeError
        
        # construct the model
        self.construct_model()

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.
        """
        logging.info(f"Attempting to build UNet architecture using cfg: {self.cfg}")
        _down_dict = OrderedDict()
        _up_dict = OrderedDict()
        # iterate over the down part
        in_channels = self.cfg['in_channels']
        for filter in self.cfg['filtrations']:
            _down_dict[f'down_filter_double_conv{filter}'] = DoubleConv2d(
                name=f'down_{filter}',
                in_channels=in_channels,
                out_channels=filter,
                kernel_size=self.cfg['double_conv_kernel'],
                stride=self.cfg['double_conv_stride'],
                dilation=self.cfg['double_conv_dilation'],
                dimension=self.cfg['double_conv_dimension'],
                activation=self.cfg['double_conv_activation'],
                batch_norm=self.cfg['double_conv_batch_norm'],
            )
            # set new in channel to current filter size
            in_channels = filter
        # iterate over the up part
        # The Convolution Transpose in ME has the following constructor arguments:
        # MinkowskiConvolutionTranspose(
        #   in_channels, 
        #   out_channels, 
        #   kernel_size=-1, 
        #   stride=1, 
        #   dilation=1, 
        #   bias=False, 
        #   kernel_generator=None, 
        #   expand_coordinates=False, 
        #   convolution_mode=<ConvolutionMode.DEFAULT: 0>, 
        #   dimension=None)
        for filter in reversed(self.cfg['filtrations']):
            _up_dict[f'up_filter_transpose{filter}'] = ME.MinkowskiConvolutionTranspose(
                in_channels=2*filter,   # adding the skip connection, so the input doubles
                out_channels=filter,
                kernel_size=self.cfg['conv_transpose_kernel'],
                stride=self.cfg['conv_transpose_stride'],
                dilation=self.cfg['conv_transpose_dilation'],
                dimension=self.cfg['conv_transpose_dimension']    
            )
            _up_dict[f'up_filter_double_conv{filter}'] = DoubleConv2d(
                name=f'up_{filter}',
                in_channels=2*filter,
                out_channels=filter,
                kernel_size=self.cfg['double_conv_kernel'],
                stride=self.cfg['double_conv_stride'],
                dilation=self.cfg['double_conv_dilation'],
                dimension=self.cfg['double_conv_dimension'],
                activation=self.cfg['double_conv_activation'],
                batch_norm=self.cfg['double_conv_batch_norm'],
            )
        # create bottleneck layer
        self.bottleneck = DoubleConv2d(
            name=f"bottleneck_{self.cfg['filtrations'][-1]}",
            in_channels=self.cfg['filtrations'][-1],
            out_channels=2*self.cfg['filtrations'][-1],
            kernel_size=self.cfg['double_conv_kernel'],
            stride=self.cfg['double_conv_stride'],
            dilation=self.cfg['double_conv_dilation'],
            dimension=self.cfg['double_conv_dimension'],
            activation=self.cfg['double_conv_activation'],
            batch_norm=self.cfg['double_conv_batch_norm'],
        )
        # create output layer
        # The Convolution layer in ME has the following constructor arguments:
        # MinkowskiConvolution(
        #   in_channels, 
        #   out_channels, 
        #   kernel_size=-1, 
        #   stride=1, 
        #   dilation=1, 
        #   bias=False, 
        #   kernel_generator=None, 
        #   expand_coordinates=False, 
        #   convolution_mode=<ConvolutionMode.DEFAULT: 0>, 
        #   dimension=None)
        self.output = ME.MinkowskiConvolution(
            in_channels=self.cfg['filtrations'][0],# to match first filtration
            out_channels=self.cfg['out_channels'], # to the number of classes
            kernel_size=1,                         # a one-one convolution
            dimension=self.cfg['double_conv_dimension'],
        )
        # create the max pooling layer
        # The Max Pooling layer from ME has the following constructor arguments:
        # MinkowskiMaxPooling(
        #   kernel_size, 
        #   stride=1, 
        #   dilation=1, 
        #   kernel_generator=None, 
        #   dimension=None)
        self.max_pooling = ME.MinkowskiMaxPooling(
            kernel_size=self.cfg['max_pooling_kernel'],
            stride=self.cfg['max_pooling_stride'],
            dilation=self.cfg['max_pooling_dilation'],
            dimension=self.cfg['max_pooling_dimension']
        )
        # create the dictionaries
        self.module_down_dict = nn.ModuleDict(_down_dict)
        self.module_up_dict = nn.ModuleDict(_up_dict)
        # record the info
        logging.info(f"Constructed UNet with down: {self.module_down_dict} and up: {self.module_up_dict}.")
        logging.info(f"Bottleneck layer: {self.bottleneck}, output layer: {self.output} and max pooling: {self.max_pooling}.")

    def forward(self, 
        x
    ):
        """
        Iterate over the module dictionary.
        """
        # record the skip connections
        skip_connections = {}
        # iterate over the down part
        for filter in self.cfg['filtrations']:
            x = self.module_down_dict[f'down_filter_double_conv{filter}'](x)
            skip_connections[f'{filter}'] = x
            x = self.max_pooling(x)
        # through the bottleneck layer
        x = self.bottleneck(x)
        for filter in reversed(self.cfg['filtrations']):
            x = self.module_up_dict[f'up_filter_transpose{filter}'](x)
            # concatenate the skip connections
            skip_connection = skip_connections[f'{filter}']
            # check for compatibility
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            concat_skip = ME.cat(skip_connection, x)
            x = self.module_up_dict[f'up_filter_double_conv{filter}'](concat_skip)
        
        return self.output(x)

if __name__ == "__main__":

    doubleconv = DoubleConv2d(
        name='test_layer',
        in_channels=1,
        out_channels=1,
        dimension=3
    )
    
    unet = UNet(
        cfg=UNet_params
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
    input = ME.SparseTensor(feats, coords)

    unet.eval()
    
    y = unet(input)
    print(y)

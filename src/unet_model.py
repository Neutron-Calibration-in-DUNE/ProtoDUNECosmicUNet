"""
Example UNet using MinkowskiEngine
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import MinkowskiEngine as ME
from collections import OrderedDict
<<<<<<< HEAD:src/unet_model.py
from datetime import datetime
import getpass
=======
>>>>>>> c39b59bbe510058963085c1c64dc70094c4f7365:unet.py
import sys
import os
import csv

from unet_logger import UNetLogger

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

class DoubleConv(ME.MinkowskiNetwork):
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
        super(DoubleConv, self).__init__(dimension)
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
<<<<<<< HEAD:src/unet_model.py
=======
        #self.logger.info(f"Creating DoubleConvLayer {self.name} with in_channels: {self.in_channels}, out_channels: {self.out_channels} and dimension: {self.dimension}.")
        #self.logger.info(f"DoubleConvLayer {self.name} has activation function: {self.activation}, bias: {self.bias} and batch_norm: {self.batch_norm}.")
        #self.logger.info(f"DoubleConvLayer {self.name} has kernel_size: {self.kernel_size}, stride: {self.stride} and dilation: {self.dilation}.")
>>>>>>> c39b59bbe510058963085c1c64dc70094c4f7365:unet.py
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
<<<<<<< HEAD:src/unet_model.py
=======
        #self.logger.info(f"Constructed DoubleConvLayer: {self.module_dict}.")
>>>>>>> c39b59bbe510058963085c1c64dc70094c4f7365:unet.py

    def forward(self, 
        x
    ):
        """
        Iterate over the module dictionary.
        """
        for layer in self.module_dict.keys():
            x = self.module_dict[layer](x)
        return x

""" 
    Here are a set of standard UNet parameters, which must be 
    adjusted by the user for each application
"""
sparse_unet_params = {
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

class SparseUNet(nn.Module):
    """
    """
    def __init__(self,
        name:   str='my_unet',      # name of the model
        cfg:    dict=sparse_unet_params    # configuration parameters
    ):
        super(SparseUNet, self).__init__()
        self.name = name
        self.logger = UNetLogger('model', file_mode='w')
        self.cfg = cfg
        # check cfg
        self.logger.info(f"checking UNet architecture using cfg: {self.cfg}")
        for item in sparse_unet_params.keys():
            if item not in self.cfg:
                self.logger.error(f"parameter {item} was not specified in config file {self.cfg}")
                raise AttributeError
        
        # construct the model
        self.construct_model()
        self.save_model(flag='init')

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.

        The Convolution Transpose in ME has the following constructor arguments:
            MinkowskiConvolutionTranspose(
            in_channels, 
            out_channels, 
            kernel_size=-1, 
            stride=1, 
            dilation=1, 
            bias=False, 
            kernel_generator=None, 
            expand_coordinates=False, 
            convolution_mode=<ConvolutionMode.DEFAULT: 0>, 
            dimension=None)

        The Convolution layer in ME has the following constructor arguments:
            MinkowskiConvolution(
            in_channels, 
            out_channels, 
            kernel_size=-1, 
            stride=1, 
            dilation=1, 
            bias=False, 
            kernel_generator=None, 
            expand_coordinates=False, 
            convolution_mode=<ConvolutionMode.DEFAULT: 0>, 
            dimension=None)

        The Max Pooling layer from ME has the following constructor arguments:
            MinkowskiMaxPooling(
            kernel_size, 
            stride=1, 
            dilation=1, 
            kernel_generator=None, 
            dimension=None)
        """
        self.logger.info(f"Attempting to build UNet architecture using cfg: {self.cfg}")
        _down_dict = OrderedDict()
        _up_dict = OrderedDict()

        # iterate over the down part
        in_channels = self.cfg['in_channels']
        for filter in self.cfg['filtrations']:
            _down_dict[f'down_filter_double_conv{filter}'] = DoubleConv(
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
        for filter in reversed(self.cfg['filtrations']):
            _up_dict[f'up_filter_transpose{filter}'] = ME.MinkowskiConvolutionTranspose(
                in_channels=2*filter,   # adding the skip connection, so the input doubles
                out_channels=filter,
                kernel_size=self.cfg['conv_transpose_kernel'],
                stride=self.cfg['conv_transpose_stride'],
                dilation=self.cfg['conv_transpose_dilation'],
                dimension=self.cfg['conv_transpose_dimension']    
            )
            _up_dict[f'up_filter_double_conv{filter}'] = DoubleConv(
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
        self.bottleneck = DoubleConv(
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
        self.output = ME.MinkowskiConvolution(
            in_channels=self.cfg['filtrations'][0],# to match first filtration
            out_channels=self.cfg['out_channels'], # to the number of classes
            kernel_size=1,                         # a one-one convolution
            dimension=self.cfg['double_conv_dimension'],
        )

        # create the max pooling layer
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
        self.logger.info(f"Constructed UNet with down: {self.module_down_dict} and up: {self.module_up_dict}.")
        self.logger.info(f"Bottleneck layer: {self.bottleneck}, output layer: {self.output} and max pooling: {self.max_pooling}.")

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
    
    def save_model(self,
        flag:   str=''
    ):
        # save meta information
        output = "models/" + self.name
        if flag != '':
            output += "_" + flag
        if not os.path.exists("models/"):
            os.makedirs("models/")
        meta_info = [[f'Meta information for model {self.name}']]
        meta_info.append(['date:',datetime.now().strftime("%m/%d/%Y %H:%M:%S")])
        meta_info.append(['user:', getpass.getuser()])
        meta_info.append(['user_id:',os.getuid()])
        system_info = self.logger.get_system_info()
        if len(system_info) > 0:
            meta_info.append(['System information:'])
            for item in system_info:
                meta_info.append([item,system_info[item]])
            meta_info.append([])
        meta_info.append(['Model configuration:'])
        meta_info.append([])
        for item in self.cfg:
            meta_info.append([item, self.cfg[item]])
        meta_info.append([])
        meta_info.append(['Model dictionary:'])
        for item in self.state_dict():
            meta_info.append([item, self.state_dict()[item].size()])
        meta_info.append([])
        with open(output + "_meta.csv", "w") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerows(meta_info)
        # save config
        cfg = [[item, self.cfg[item]] for item in self.cfg]
        with open(output+".cfg", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(cfg)
        # save parameters
        torch.save(self.state_dict(), output + "_params.ckpt")

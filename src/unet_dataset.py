"""
Class for constructing UNet datasets
"""
# imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import sys
import math
import csv
from unet_logger import UNetLogger
from sklearn import cluster
from sklearn import metrics
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import MinkowskiEngine as ME

class NeutronUNetDataset(Dataset):
    """
    """
    def __init__(self,
        input_file: str,
    ):
        self.logger = UNetLogger('dataset', file_mode='w')
        self.logger.info(f"Attempting to construct UNet dataset from {input_file}.")
        super(NeutronUNetDataset, self).__init__()
        self.input_file = np.load(input_file, allow_pickle=True)
        self.coords = self.input_file['coords']
        self.feats  = self.input_file['feats']
        self.labels = self.input_file['labels']
        self.num_events = len(self.coords)
        self.weights = None

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        coords = np.array(self.coords[idx])
        feats  = np.array(self.feats[idx])
        labels = np.array(self.labels[idx])
        return coords, feats, labels
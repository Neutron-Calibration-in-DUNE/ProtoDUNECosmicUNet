"""
Example semantic segmentation clustering analysis
"""
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torch.nn as nn
import sys
sys.path.append('src/')

from unet_model import SparseUNet
from unet_dataset import NeutronUNetDataset
from unet_loader import SparseUNetLoader
from unet_metrics import SparseBinarySemanticSegmentationMetrics
from unet_losses import GenericLoss
from unet_optimizers import GenericOptimizer
from unet_trainer import SparseTrainer
from unet_analysis import UNetAnalyzer
import numpy as np

# analyze the results
cosmic_analyzer = UNetAnalyzer(
    input_file='predictions/protodune_cosmic_voxels_predictions.npz'
)
cosmic_analyzer.plot_predictions(5)
cosmic_analyzer.cluster()
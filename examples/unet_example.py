"""
Example semantic segmentation using the melange API
"""
import os
from PIL import Image
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


cosmic_dataset = NeutronUNetDataset(
    "../neutron_data/unet_dataset.npz",
)

cosmic_loader = SparseUNetLoader(
    cosmic_dataset, 
    batch_size=5,
    num_workers=1,
    validation_split=0.3,
)

# create unet model with default parameters
cosmic_unet = SparseUNet(
    name='neutron_cosmic_unet'
)

# algorithm (Adam)
cosmic_loss = GenericLoss('BCEWithLogitsLoss')
cosmic_optimizer = GenericOptimizer(
    cosmic_unet, 
    'Adam'
)
cosmic_metrics = SparseBinarySemanticSegmentationMetrics()

# trainer
cosmic_trainer = SparseTrainer(
    cosmic_unet,
    cosmic_loss,
    cosmic_optimizer,
    metrics=cosmic_metrics,
    gpu=True,
    gpu_device=0,
    semantic_segmentation=True
)

cosmic_trainer.train(
    cosmic_loader,
    epochs=50,
)

cosmic_unet.save_model(
    flag='trained'
)

cosmic_trainer.inference(
    cosmic_loader,
    output_file='unet_predictions.npz'
)

cosmic_analyzer = UNetAnalyzer(
    input_file='predictions/unet_predictions.npz'
)
cosmic_analyzer.plot_predictions(5)

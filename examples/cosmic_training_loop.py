"""
Example semantic segmentation using the melange API
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

# first we construct the model, loss, optimizer, metrics and trainer
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

# define training and testing sets
training_dir = "training/"
training_files = [
    "protodune_cosmic_voxels_0.npz",
    "protodune_cosmic_voxels_1.npz",
    "protodune_cosmic_voxels_2.npz",
]
testing_dir = "testing/"
testing_file = "protodune_cosmic_voxels_3.npz"

# now iterate over training datasets.
for training_file in training_files:

    train_dataset = NeutronUNetDataset(
        training_dir + training_file,
    )

    train_loader = SparseUNetLoader(
        train_dataset, 
        batch_size=5,
        num_workers=1,
        validation_split=0.0,
        test_split=0.0,
    )
    # train 
    cosmic_trainer.train(
        train_loader,
        epochs=50,
    )

# save the final model
cosmic_unet.save_model(
    flag='trained'
)

# conduct inference on the test set
test_dataset = NeutronUNetDataset(
    testing_dir + testing_file,
)

test_loader = SparseUNetLoader(
    test_dataset, 
    batch_size=5,
    num_workers=1,
    validation_split=0.0,
    test_split=0.0,
)
cosmic_trainer.inference(
    test_loader,
    output_file='protodune_cosmic_voxels_predictions.npz'
)
"""
Script for generating neutron datasets
"""
# imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../src/")
import csv
from neutron_dataset_new import NeutronCosmicDataset

"""
Generate training datasets from root files.
"""
training_dir = "../../neutron_data/"
training_output = "../sparse_unet/training/"
if not os.path.isdir(training_output):
    os.mkdir(training_output)

training_files = [
    "protodune_cosmic_voxels_0",
    "protodune_cosmic_voxels_1",
    "protodune_cosmic_voxels_2"
]
# iterate over the training files
for training_file in training_files:
    # load each root file and construct numpy files
    dataset = NeutronCosmicDataset(
        name=f'train_{training_file}',
        input_file=training_dir + training_file + ".root",
        load_neutrons=True,
        load_mc_edeps=True,
        load_mc_voxels=True,
        load_reco_edeps=False,
    )
    dataset.generate_unet_training(
        output_file=training_output + training_file + ".npz"
    )

"""
Generate testing dataset from root file.
"""
# create test set
testing_dir = training_dir

testing_output = "../sparse_unet/testing/"
if not os.path.isdir(testing_output):
    os.mkdir(testing_output)

testing_file = "protodune_cosmic_voxels_0"

# create the testing set
dataset = NeutronCosmicDataset(
    name=f'test_{testing_file}',
    input_file=testing_dir + testing_file + ".root",
    load_neutrons=True,
    load_mc_edeps=True,
    load_mc_voxels=True,
    load_reco_edeps=False,
)
dataset.generate_unet_training(
    output_file=testing_output + testing_file + ".npz"
)
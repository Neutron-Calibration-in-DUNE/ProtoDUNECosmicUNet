"""
Example using NeutronDataset
"""
# imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import sys
import math
import csv
sys.path.append("../src/")
from neutron_dataset import NeutronCosmicDataset

# create a NeutronCosmicDataset from an input root file
# data from LArSoft is stored one directory up from the
# root directory.
input_dir   = "../../neutron_data/"
input_file  = "protodune_cosmic_voxels_0.root"
dataset = NeutronCosmicDataset(
    input_file      = input_dir + input_file,
    load_neutrons   = True,     # loads the neutron TTree
    load_muons      = True,     # loads the muon TTree
    load_ar39       = False,    # loads the ar39 TTree
    load_voxels     = True,     # loads the voxel TTree
)

# Various plotting functions are shown below.
# results are saved in a 'plots/' folder.
dataset.plot_event(
    event   = 10,
    title   = 'ProtoDUNE Cosmic Example',
    show_active_tpc = True,     # shows outline of active LAr
    show_cryostat   = True,     # shows outline of cryostat
    save    = 'neutron_cosmic_event_example',   # name of the file to save the plot as
    show    = False,
)
# plot neutron captures labeled/colored by 
# the neutron ID
dataset.plot_event_neutrons(
    event   = 10, 
    label   = 'neutron',    
    title   = 'ProtoDUNE Example Capture by Neutron ID',
    legend_cutoff   = 5,        # show the first N labels in the legend
    show_active_tpc = True,     # shows outline of active LAr
    show_cryostat   = True,     # shows outline of cryostat
    save    = 'neutron_cosmic_neutron_example',  # name of the file to save the plot as
    show    = False,
)
# plot neutron captures labeled/colored by 
# the gamma ID
dataset.plot_event_neutrons(
    event   = 10, 
    label   = 'gamma',    
    title   = 'ProtoDUNE Example Capture by Gamma ID',
    legend_cutoff   = 5,        # show the first N labels in the legend
    show_active_tpc = True,     # shows outline of active LAr
    show_cryostat   = True,     # shows outline of cryostat
    save    = 'neutron_cosmic_gamma_example',  # name of the file to save the plot as
    show    = False,
)
# plot 3d and xz/xy views of capture locations
dataset.plot_capture_locations(
    event       = 0,
    plot_type   = '3d',
    show_active_tpc =True,
    show_cryostat   =True,
    title       = 'Example Capture Locations (ProtoDUNE)',
    save        = 'neutron_captures_protodune_3d',
    show        = False
)
dataset.plot_capture_locations(
    event       = 0,
    plot_type   = 'xz',
    show_active_tpc =True,
    show_cryostat   =True,
    title       = 'Example Capture Locations (ProtoDUNE)',
    save        = 'neutron_captures_protodune_xz',
    show        = False
)
dataset.plot_capture_locations(
    event       = 0,
    plot_type   = 'xy',
    show_active_tpc =True,
    show_cryostat   =True,
    title       = 'Example Capture Locations (ProtoDUNE)',
    save        = 'neutron_captures_protodune_xy',
    show        = False
)
# plot the density of captures along a particular
# 2d plane
dataset.plot_capture_density(
    plot_type   = 'xy',
    density_type= 'kde',
    title       = 'Example Capture Location Density (ProtoDUNE)',
    save        = 'neutron_capture_density_xy',
    show        = False,
)
# fit an exponential to the histogram of captures along the
# y direction.
dataset.fit_depth_exponential(
    num_bins=100,
    save    ='neutron_cosmic_depth_exponential',
    show    = False,
)

# generate a SparseUNet training set
dataset.generate_unet_training(
    output_file="../../neutron_data/unet_dataset.npz",
)

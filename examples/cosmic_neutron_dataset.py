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
input_file  = "protodune_cosmic_voxels_0"
dataset = NeutronCosmicDataset(
    name            = input_file,
    input_file      = input_dir + input_file + ".root",
    load_neutrons   = True,
    load_mc_edeps   = True,
    load_mc_voxels  = True,
    load_reco_edeps = False,
)
# Various plotting functions are shown below.
for event in range(dataset.num_events):
    # # results are saved in a 'plots/' folder.
    # dataset.plot_event(
    #     event   = event,
    #     title   = 'ProtoDUNE Cosmic Example',
    #     show_active_tpc = True,     # shows outline of active LAr
    #     show_cryostat   = True,     # shows outline of cryostat
    # )
    # # plot neutron captures labeled/colored by 
    # # the neutron ID
    # dataset.plot_event_neutrons(
    #     event   = event, 
    #     label   = 'neutron',    
    #     title   = 'ProtoDUNE Example Capture by Neutron ID',
    #     legend_cutoff   = 5,        # show the first N labels in the legend
    #     show_active_tpc = True,     # shows outline of active LAr
    #     show_cryostat   = True,     # shows outline of cryostat
    # )
    # # plot neutron captures labeled/colored by 
    # # the gamma ID
    # dataset.plot_event_neutrons(
    #     event   = event, 
    #     label   = 'gamma',    
    #     title   = 'ProtoDUNE Example Capture by Gamma ID',
    #     legend_cutoff   = 5,        # show the first N labels in the legend
    #     show_active_tpc = True,     # shows outline of active LAr
    #     show_cryostat   = True,     # shows outline of cryostat
    #     save    = 'neutron_cosmic_gamma_example',  # name of the file to save the plot as
    #     show    = False,
    # )
    # plot 3d and xz/xy views of capture locations
    dataset.plot_capture_locations(
        event           = event,
        plot_type       = '3d',
        show_active_tpc = True,
        show_cryostat   = True,
        title           = 'Example Capture Locations (ProtoDUNE)',
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

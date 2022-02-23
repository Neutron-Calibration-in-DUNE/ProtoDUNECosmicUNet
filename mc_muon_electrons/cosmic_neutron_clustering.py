"""
Example using NeutronClustering
"""
# imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import sys
import math
import csv
sys.path.append("../src/clustering")
sys.path.append("../src/")
from neutron_clustering import NeutronClustering

# create a NeutronCosmicDataset from an input root file
# data from LArSoft is stored one directory up from the
# root directory.
input_dir   = "../../neutron_data/protodune_cosmic/mc_muon_electrons"
input_file  = "protodune_cosmic_voxels_2.root"
neutron_clustering = NeutronClustering(
    input_file = input_dir + input_file
)

# cluster on neutron truth data
neutron_clustering.cluster_truth(
    level   ='neutron',
    alg     ='dbscan',
    params  ={'eps': 30.,'min_samples': 6},
)
# calculate clustering scores
avg_scores = neutron_clustering.calc_truth_scores(
    level='neutron',
)
print(f"clustering scores: ")
for item in avg_scores.keys(): 
    print(f"    - {item}: {avg_scores[item]}.")

# plot the true and predicted energy spectrums
neutron_clustering.plot_true_spectrum(
    num_bins    = 100,
    energy_cut  = 10.,
    title       = 'Neutron Clustering True Energy Spectrum (ProtoDUNE)',
    save        = 'neutron_clustering_true_spectrum',
    show        = False
)
neutron_clustering.plot_prediction_spectrum(
    num_bins    = 100,
    edep_type   = 'compare',
    energy_cut  = 10.,
    title       = 'Neutron Clustering Prediction Spectrum (ProtoDUNE)',
    save        = 'neutron_clustering_prediction_spectrum',
    show        = False
)

# this section attempts to optimize the hyperparameters
# for the dbscan clustering algorithm by scanning
# a range of eps values for dbscan on truth data.
eps_range = [1., 100.]
eps_step  = 1.
run_scan  = True
if run_scan:
    neutron_clustering.scan_truth_eps_values(
        eps_range   = eps_range,
        eps_step    = eps_step
    )
    # plot the results of the scan scores
    neutron_clustering.plot_truth_eps_scores(
        input_file  = "scan_scores.csv",
        save        = 'scan_scores'
    )
"""
Collection of parameters for various classes
"""
# imports
import os
import sys
import math
import csv

"""
The "neutron" array should have the following entries:
    event_id:           the event id for each event (e.g. [0, 7, 18, ...])
    neutron_ids:        track id of each neutron in the event (e.g. [1, 2, 3, ...])
    neutron_capture_x:  the x position of each neutron capture (e.g. [54, 154, ...])
    neutron_capture_y:  the y position ""
    neutron_capture_z:  the z position ""
    gamma_ids:          track id of each gamma that comes from a neutron capture (e.g. [65, 66, ...])
    gamma_neutron_ids:  the track id of the parent of each gamma (e.g. [1, 1, 1, 2, 2, ...])
    gamma_energy (GeV): the energy values of each unique gamma in the event (e.g. [0.004745, ...])
    edep_energy:        the energy values for each deposited energy from a gamma (e.g. [0.00038, ...])
    edep_parent:        the track id of the particle which left the energy deposit ^^^
    edep_neutron_ids:   the track id of the neutron which led to the energy deposit ^^^
    edep_gamma_ids:     the track id of the gamma which left behind each edep in "edep_energy" (e.g. [65, 65, 65, ...])
    edep_x (mm):        the x position of each edep in the event (e.g. [-42, 500.1, ...])
    edep_y (mm):        the y position ""
    edep_z (mm):        the z position ""
    electron_ids:           the track id of each electron tracked in the simulation that comes from a gamma
    electron_neutron_ids:   the corresponding id of the neutron that generated the electron with id ^^^
    electron_gamma_ids:     the corresponding id of the gamma that generated the electron with id ^^^
    electron_energy (GeV):  the energy of each electron tracked in the simulation (e.g. [0.00058097, ...])
    edep_num_electrons:     the number of electrons coming out of the IonAndScint simulation for each edep ^^^
""" 
required_neutron_arrays = [
    'event_id',
    'neutron_ids',
    'neutron_capture_x',
    'neutron_capture_y',
    'neutron_capture_z',
    'gamma_ids',
    'gamma_neutron_ids',
    'gamma_energy',
    'edep_energy',
    'edep_parent',
    'edep_neutron_ids',
    'edep_gamma_ids',
    'edep_x',
    'edep_y',
    'edep_z',
    'electron_ids',
    'electron_neutron_ids',
    'electron_gamma_ids',
    'electron_energy',
    'edep_num_electrons',
]

""" 
The "muon" array should have the following entries:
    primary_muons:      the number of muons in the event
    muon_ids:           the track ids of the muons
    muon_edep_ids:      the track id of the corresponding muon that left the energy deposit
    muon_edep_energy:   the energy values of each unique deposit
    muon_edep_num_electrons:    the number of electrons generated from each energy deposit
    muon_edep_x:        the x position of each edep from muons
    muon_edep_y:        the y ""
    muon_edep_z:        the z ""
"""

required_muon_arrays = [
    'primary_muons',
    'muon_ids',
    'muon_edep_ids',
    'muon_edep_energy',
    'muon_edep_num_electrons',
    'muon_edep_x',
    'muon_edep_y',
    'muon_edep_z',
]

required_voxel_arrays = [
    "x_min", 
    "x_max", 
    "y_min", 
    "y_max", 
    "z_min", 
    "z_max", 
    "voxel_size", 
    "num_voxels_x",
    "num_voxels_y",
    "num_voxels_z",
    "x_id", 
    "y_id", 
    "z_id", 
    "values",
    "labels",
]

cluster_params = {
    'affinity':     {'damping': 0.5, 'max_iter': 200},
    'mean_shift':   {'bandwidth': None},
    'dbscan':       {'eps': 100.,'min_samples': 6},
    'optics':       {'min_samples': 6},
    'gaussian':     {'n_components': 1, 'covariance_type': 'full', 'tol': 1e-3, 'reg_covar': 1e-6, 'max_iter': 100}
}

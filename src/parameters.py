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
    'neutron_ids',
    'primary',
    'capture',
    'capture_tpc',
    'capture_tpc_lar',
    'inelastic',
    'total_number_steps',
    'cryo_number_steps',
    'tpc_number_steps',
    'lar_number_steps',
    'entered_tpc',
    'entered_tpc_step',
    'entered_tpc_time',
    'entered_tpc_energy',
    'exited_tpc',
    'exited_tpc_step',
    'exited_tpc_time',
    'exited_tpc_energy',
    'tpc_avg_material',
    'total_distance',
    'cryo_distance',
    'tpc_distance',
    'neutron_capture_x',
    'neutron_capture_y',
    'neutron_capture_z',
]

required_mc_edep_arrays = [
    'pdg',
    'track_id',
    'ancestor_id',
    'level',
    'edep_x',
    'edep_y',
    'edep_z',
    'energy',
    'num_electrons',
]

required_voxel_arrays = [
    'voxels',
    'labels',
    'energy',
    'edep_idxs',
]

required_reco_edep_arrays = [
    'pdg',
    'track_id',
    'ancestor_id',
    'level',
    'sp_x',
    'sp_y',
    'sp_z',
    'summed_adc',
]

cluster_params = {
    'affinity':     {'damping': 0.5, 'max_iter': 200},
    'mean_shift':   {'bandwidth': None},
    'dbscan':       {'eps': 100.,'min_samples': 6},
    'optics':       {'min_samples': 6},
    'gaussian':     {'n_components': 1, 'covariance_type': 'full', 'tol': 1e-3, 'reg_covar': 1e-6, 'max_iter': 100}
}

pdg_map = {
    '11':   'electron',
    '13':   'muon',
    '22':   'gamma',
    '2112': 'neutron',
    '2212': 'proton'
}

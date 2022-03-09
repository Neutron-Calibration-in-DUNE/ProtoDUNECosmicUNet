"""
Collection of classes for generating cosmic ray datasets
for training and clustering from LArSoft output.
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
from sklearn import cluster
from sklearn import metrics
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from unet_logger import UNetLogger
from parameters import *

class NeutronCosmicDataset:
    """
    This class loads simulated neutron events and runs various clustering
    algorithms and analysis.  The arrays in the root file should be structured
    as follows:
        meta:       meta information such as ...
        geometry:   information about the detector geometry such as volume bounding boxes...
        neutron:    the collection of neutron event information from the simulation
        muon:       the collection of muon event information
        voxels:     the collection of voxelized truth/reco information
    """
    def __init__(self,
        name:       str,
        input_file: str,
        load_neutrons:  bool=True,
        load_mc_edeps:  bool=True,
        load_mc_voxels: bool=True,
        load_reco_edeps:bool=True,
    ):
        self.load_neutrons  = load_neutrons
        self.load_mc_edeps  = load_mc_edeps
        self.load_mc_voxels = load_mc_voxels
        self.load_reco_edeps= load_reco_edeps
        self.name = name
        self.logger = UNetLogger('neutron_dataset', file_mode='w')
        self.logger.info(f"Attempting to load file {input_file}.")
        # load the file
        try:
            self.input_file = uproot.open(input_file)
            self.logger.info(f"Successfully loaded file {input_file}.")
        except Exception:
            self.logger.error(f"Failed to load file with exception: {Exception}.")
            raise Exception
        if not os.path.isdir(f"plots/{self.name}/events/"):
            os.makedirs(f"plots/{self.name}/events/")
        # now load the various arrays
        self.meta = self.load_array(self.input_file, 'ana/meta')
        self.geometry = self.load_array(self.input_file, 'ana/geometry')
        if load_neutrons:
            self.neutron = self.load_array(self.input_file, 'ana/mc_neutron_captures')
            for item in required_neutron_arrays:
                if item not in self.neutron.keys():
                    self.logger.info(f"Required array {item} not present in mc_neutron_captures!")
                    raise ValueError(f"Required array {item} not present in mc_neutron_captures!")
            self.num_neutron_events = len(self.neutron['neutron_capture_x'])
            self.logger.info(f"Loaded 'neutron' arrays with {self.num_neutron_events} entries.")
        if load_mc_edeps:
            self.mc_edeps = self.load_array(self.input_file, 'ana/mc_energy_deposits')
            for item in required_mc_edep_arrays:
                if item not in self.mc_edeps.keys():
                    self.logger.info(f"Required array {item} not present in mc_energy_deposits!")
                    raise ValueError(f"Required array {item} not present in mc_energy_deposits!")
            self.num_mc_edep_events = len(self.mc_edeps['pdg'])
            self.logger.info(f"Loaded 'mc_energy_deposits' arrays with {self.num_mc_edep_events} entries.")
        if load_mc_voxels:
            self.mc_voxels = self.load_array(self.input_file, 'ana/mc_voxels')
            for item in required_voxel_arrays:
                if item not in self.mc_voxels.keys():
                    self.logger.info(f"Required array {item} not present in mc_voxels!")
                    raise ValueError(f"Required array {item} not present in mc_voxels!")
            self.num_mc_voxel_events = len(self.mc_voxels['voxels'])
            self.discrete_voxel_values = [
                [[1.] for i in range(len(self.mc_voxels['voxels'][i]))] 
                    for i in range(len(self.mc_voxels['voxels']))
                ]
            self.logger.info(f"Loaded 'mc_voxels' arrays with {self.num_mc_voxel_events} entries.")
        if load_reco_edeps:
            self.reco_edeps = self.load_array(self.input_file, 'ana/reco_energy_deposits')

            
        # construct TPC boxes
        self.world_box_ranges = self.geometry['world_box_ranges']
        self.world_x = [self.world_box_ranges[0][0], self.world_box_ranges[0][1]]
        self.world_y = [self.world_box_ranges[0][4], self.world_box_ranges[0][5]]
        self.world_z = [self.world_box_ranges[0][2], self.world_box_ranges[0][3]]
        self.total_tpc_ranges = self.geometry['total_active_tpc_box_ranges']
        self.tpc_x = [self.total_tpc_ranges[0][0], self.total_tpc_ranges[0][1]]
        self.tpc_y = [self.total_tpc_ranges[0][2], self.total_tpc_ranges[0][3]]
        self.tpc_z = [self.total_tpc_ranges[0][4], self.total_tpc_ranges[0][5]]
        self.active_tpc_lines = [
            [[self.tpc_x[0],self.tpc_y[0],self.tpc_z[0]],[self.tpc_x[1],self.tpc_y[0],self.tpc_z[0]]],
            [[self.tpc_x[0],self.tpc_y[0],self.tpc_z[0]],[self.tpc_x[0],self.tpc_y[1],self.tpc_z[0]]],
            [[self.tpc_x[0],self.tpc_y[0],self.tpc_z[0]],[self.tpc_x[0],self.tpc_y[0],self.tpc_z[1]]],
            [[self.tpc_x[0],self.tpc_y[1],self.tpc_z[0]],[self.tpc_x[1],self.tpc_y[1],self.tpc_z[0]]],
            [[self.tpc_x[0],self.tpc_y[1],self.tpc_z[0]],[self.tpc_x[0],self.tpc_y[1],self.tpc_z[1]]],
            [[self.tpc_x[1],self.tpc_y[0],self.tpc_z[0]],[self.tpc_x[1],self.tpc_y[0],self.tpc_z[1]]],
            [[self.tpc_x[1],self.tpc_y[0],self.tpc_z[0]],[self.tpc_x[1],self.tpc_y[1],self.tpc_z[0]]],
            [[self.tpc_x[0],self.tpc_y[1],self.tpc_z[1]],[self.tpc_x[1],self.tpc_y[1],self.tpc_z[1]]],
            [[self.tpc_x[0],self.tpc_y[1],self.tpc_z[1]],[self.tpc_x[0],self.tpc_y[0],self.tpc_z[1]]],
            [[self.tpc_x[1],self.tpc_y[0],self.tpc_z[1]],[self.tpc_x[1],self.tpc_y[1],self.tpc_z[1]]],
            [[self.tpc_x[1],self.tpc_y[0],self.tpc_z[1]],[self.tpc_x[0],self.tpc_y[0],self.tpc_z[1]]],
            [[self.tpc_x[1],self.tpc_y[1],self.tpc_z[0]],[self.tpc_x[1],self.tpc_y[1],self.tpc_z[1]]],
        ]
        # cryostat boundary
        self.total_cryo_ranges = self.geometry['cryostat_box_ranges']
        self.cryo_x = [self.total_cryo_ranges[0][0], self.total_cryo_ranges[0][1]]
        self.cryo_y = [self.total_cryo_ranges[0][2], self.total_cryo_ranges[0][3]]
        self.cryo_z = [self.total_cryo_ranges[0][4], self.total_cryo_ranges[0][5]]
        self.cryostat_lines = [
            [[self.cryo_x[0],self.cryo_y[0],self.cryo_z[0]],[self.cryo_x[1],self.cryo_y[0],self.cryo_z[0]]],
            [[self.cryo_x[0],self.cryo_y[0],self.cryo_z[0]],[self.cryo_x[0],self.cryo_y[1],self.cryo_z[0]]],
            [[self.cryo_x[0],self.cryo_y[0],self.cryo_z[0]],[self.cryo_x[0],self.cryo_y[0],self.cryo_z[1]]],
            [[self.cryo_x[0],self.cryo_y[1],self.cryo_z[0]],[self.cryo_x[1],self.cryo_y[1],self.cryo_z[0]]],
            [[self.cryo_x[0],self.cryo_y[1],self.cryo_z[0]],[self.cryo_x[0],self.cryo_y[1],self.cryo_z[1]]],
            [[self.cryo_x[1],self.cryo_y[0],self.cryo_z[0]],[self.cryo_x[1],self.cryo_y[0],self.cryo_z[1]]],
            [[self.cryo_x[1],self.cryo_y[0],self.cryo_z[0]],[self.cryo_x[1],self.cryo_y[1],self.cryo_z[0]]],
            [[self.cryo_x[0],self.cryo_y[1],self.cryo_z[1]],[self.cryo_x[1],self.cryo_y[1],self.cryo_z[1]]],
            [[self.cryo_x[0],self.cryo_y[1],self.cryo_z[1]],[self.cryo_x[0],self.cryo_y[0],self.cryo_z[1]]],
            [[self.cryo_x[1],self.cryo_y[0],self.cryo_z[1]],[self.cryo_x[1],self.cryo_y[1],self.cryo_z[1]]],
            [[self.cryo_x[1],self.cryo_y[0],self.cryo_z[1]],[self.cryo_x[0],self.cryo_y[0],self.cryo_z[1]]],
            [[self.cryo_x[1],self.cryo_y[1],self.cryo_z[0]],[self.cryo_x[1],self.cryo_y[1],self.cryo_z[1]]],
        ]
        self.calculate_capture_ratio()

    def load_array(self,
        input_file,
        array_name,
    ):
        self.logger.info(f"Attempting to load array: {array_name} from file: {input_file}.")
        try:
            array = input_file[array_name].arrays(library="np")
            self.logger.info(f"Successfully loaded array: {array_name} from file: {input_file}.")
        except Exception:
            self.logger.error(f"Failed to load array: {array_name} from file: {input_file} with exception: {Exception}.")
            raise Exception
        return array

    """
    The following functions are for plotting and analyzing
    neutron captures, whose data products are stored in 
    self.neutron.
    """
    def plot_capture_locations(self,
        event:          int,
        plot_type:      str='3d',
        capture_location:str='tpc',
        show_active_tpc:bool=True,
        show_cryostat:  bool=True,
        title:          str='Example MC Capture Locations',
        save:           bool=True,
        show:           bool=False,
    ):
        """
        
        """
        if self.load_neutrons == False:
            self.logger.error(f"Dataset does not have 'neutron' products loaded! (i.e. 'self.load_neutrons' = {self.load_neutrons})")
            raise ValueError(f"Dataset does not have 'neutron' products loaded! (i.e. 'self.load_neutrons' = {self.load_neutrons})")
        if event >= self.num_neutron_events:
            self.logger.error(f"Tried accessing element {event} of array with size {self.num_neutron_events}!")
            raise IndexError(f"Tried accessing element {event} of array with size {self.num_neutron_events}!")
        if plot_type not in ['3d', 'xy', 'xz', 'yz']:
            self.logger.warning(f"Requested plot type '{plot_type}' not allowed, using '3d'.")
            plot_type = '3d'
        if capture_location not in ['world', 'cryostat', 'tpc']:
            self.logger.warning(f"Requested capture location '{capture_location}' not allowed, using 'tpc'.")
            capture_location = 'tpc'
        # gather x, y, z values
        x = self.neutron['neutron_capture_x'][event]
        y = self.neutron['neutron_capture_y'][event]
        z = self.neutron['neutron_capture_z'][event]
        num_neutrons = len(x)
        if capture_location == 'world':
            mask = (
                (x < self.world_x[1]) & (x > self.world_x[0]) &
                (y < self.world_y[1]) & (y > self.world_y[0]) &
                (z < self.world_z[1]) & (z > self.world_z[0])
            )
        elif capture_location == 'cryostat':
            mask = (
                (x < self.cryo_x[1]) & (x > self.cryo_x[0]) &
                (y < self.cryo_y[1]) & (y > self.cryo_y[0]) &
                (z < self.cryo_z[1]) & (z > self.cryo_z[0])
            )
        else:
            mask = (
                (x < self.tpc_x[1]) & (x > self.tpc_x[0]) &
                (y < self.tpc_y[1]) & (y > self.tpc_y[0]) &
                (z < self.tpc_z[1]) & (z > self.tpc_z[0])
            )
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if plot_type == '3d':
            fig = plt.figure(figsize=(8,6))
            axs = fig.add_subplot(projection='3d')
            axs.scatter(x, z, y)
            axs.set_xlabel("x (mm)")
            axs.set_ylabel("z (mm)")
            axs.set_zlabel("y (mm)")
            # draw the active tpc volume box
        else:
            fig, axs = plt.subplots(figsize=(8,6))
            if plot_type == 'xz':
                axs.scatter(x, z)
                axs.set_xlabel("x (mm)")
                axs.set_ylabel("z (mm)")
            elif plot_type == 'yz':
                axs.scatter(y, z)
                axs.set_xlabel("y (mm)")
                axs.set_ylabel("z (mm)")
            else:
                axs.scatter(x, y)
                axs.set_xlabel("x (mm)")
                axs.set_ylabel("y (mm)")
        if show_active_tpc:
            for i in range(len(self.active_tpc_lines)):
                x = np.array([
                    self.active_tpc_lines[i][0][0],
                    self.active_tpc_lines[i][1][0]
                ])
                y = np.array([
                    self.active_tpc_lines[i][0][1],
                    self.active_tpc_lines[i][1][1]
                ])
                z = np.array([
                    self.active_tpc_lines[i][0][2],
                    self.active_tpc_lines[i][1][2]
                ])
                if plot_type == '3d':
                    if i == 0:
                        axs.plot(
                            x,z,y,
                            linestyle='--',color='b',
                            label='Active TPC volume'
                        )
                    else:
                        axs.plot(x,z,y,linestyle='--',color='b')
                elif plot_type == 'xz':
                    if i == 0:
                        axs.plot(
                            x,z,
                            linestyle='--',color='b',
                            label='Active TPC volume'
                        )
                    else:
                        axs.plot(x,z,linestyle='--',color='b')
                elif plot_type == 'yz':
                    if i == 0:
                        axs.plot(
                            y,z,
                            linestyle='--',color='b',
                            label='Active TPC volume'
                        )
                    else:
                        axs.plot(y,z,linestyle='--',color='b')
                else:
                    if i == 0:
                        axs.plot(
                            x,y,
                            linestyle='--',color='b',
                            label='Active TPC volume'
                        )
                    else:
                        axs.plot(x,y,linestyle='--',color='b')
        if show_cryostat:
            for i in range(len(self.cryostat_lines)):
                x = np.array([
                    self.cryostat_lines[i][0][0],
                    self.cryostat_lines[i][1][0]
                ])
                y = np.array([
                    self.cryostat_lines[i][0][1],
                    self.cryostat_lines[i][1][1]
                ])
                z = np.array([
                    self.cryostat_lines[i][0][2],
                    self.cryostat_lines[i][1][2]
                ])
                if plot_type == '3d':
                    if i == 0:
                        axs.plot(
                            x,z,y,
                            linestyle=':',color='g',
                            label='Cryostat volume'
                        )
                    else:
                        axs.plot(x,z,y,linestyle=':',color='g')
                elif plot_type == 'xz':
                    if i == 0:
                        axs.plot(
                            x,z,
                            linestyle=':',color='g',
                            label='Cryostat volume'
                        )
                    else:
                        axs.plot(x,z,linestyle=':',color='g')
                elif plot_type == 'yz':
                    if i == 0:
                        axs.plot(
                            y,z,
                            linestyle=':',color='g',
                            label='Cryostat volume'
                        )
                    else:
                        axs.plot(y,z,linestyle=':',color='g')
                else:
                    if i == 0:
                        axs.plot(
                            x,y,
                            linestyle=':',color='g',
                            label='Cryostat volume'
                        )
                    else:
                        axs.plot(x,y,linestyle=':',color='g')
        axs.set_title(title)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f'plots/{self.name}/events/capture_{plot_type}_{event}.png')
        if show:
            plt.show()

    def calculate_capture_ratio(self):
        # keep track of the ratio of complete 6.098
        # captures vs total.
        if self.load_mc_edeps == False:
            return
        self.logger.info(f"Attempting to calculate capture ratio.")
        self.num_complete_captures = []
        self.num_captures = []
        # loop through all edeps
        for ii, pdgs in enumerate(self.mc_edeps['pdg']):
            complete = 0
            total = 0
            truth = self.mc_edeps['ancestor_id'][ii][(pdgs == 2112)]
            clusters = np.unique(truth)
            for cluster in clusters:
                indices = np.where(truth == cluster)
                true_energies = sum(self.mc_edeps['energy'][ii][indices])
                total += 1
                if round(true_energies,2) == 6.1:
                    complete += 1
            self.num_captures.append(total)
            self.num_complete_captures.append(complete)
        self.capture_ratio = round((sum(self.num_complete_captures)/sum(self.num_captures))*100)
    
    def plot_mc_edep_locations(self,
        event,
        plot_type:      str='3d',
        capture_location:str='tpc',
        show_active_tpc:bool=True,
        show_cryostat:  bool=True,
        title:          str='Example MC Edep Locations',
        save:           bool=True,
        show:           bool=False,
    ):
        if self.load_mc_edeps == False:
            self.logger.error(f"Dataset does not have 'mc_energy_deposits' products loaded! (i.e. 'self.load_mc_edeps' = {self.load_mc_edeps})")
            raise ValueError(f"Dataset does not have 'mc_energy_deposits' products loaded! (i.e. 'self.load_mc_edeps' = {self.load_mc_edeps})")
        if event >= self.num_mc_edep_events:
            self.logger.error(f"Tried accessing element {event} of array with size {self.num_mc_edep_events}!")
            raise IndexError(f"Tried accessing element {event} of array with size {self.num_mc_edep_events}!")
        if plot_type not in ['3d', 'xy', 'xz', 'yz']:
            self.logger.warning(f"Requested plot type '{plot_type}' not allowed, using '3d'.")
            plot_type = '3d'
        if capture_location not in ['world', 'cryostat', 'tpc']:
            self.logger.warning(f"Requested capture location '{capture_location}' not allowed, using 'tpc'.")
            capture_location = 'tpc'
        # gather x, y, z values
        x = self.mc_edeps['edep_x'][event]
        y = self.mc_edeps['edep_y'][event]
        z = self.mc_edeps['edep_z'][event]
        if capture_location == 'world':
            mask = (
                (x < self.world_x[1]) & (x > self.world_x[0]) &
                (y < self.world_y[1]) & (y > self.world_y[0]) &
                (z < self.world_z[1]) & (z > self.world_z[0])
            )
        elif capture_location == 'cryostat':
            mask = (
                (x < self.cryo_x[1]) & (x > self.cryo_x[0]) &
                (y < self.cryo_y[1]) & (y > self.cryo_y[0]) &
                (z < self.cryo_z[1]) & (z > self.cryo_z[0])
            )
        else:
            mask = (
                (x < self.tpc_x[1]) & (x > self.tpc_x[0]) &
                (y < self.tpc_y[1]) & (y > self.tpc_y[0]) &
                (z < self.tpc_z[1]) & (z > self.tpc_z[0])
            )
        x = x[mask]
        y = y[mask]
        z = z[mask]
        energy = self.mc_edeps['energy'][event][mask]
        pdg = self.mc_edeps['pdg'][event][mask]
        unique_pdgs = np.unique(pdg)
        unique_x = []
        unique_y = []
        unique_z = []
        unique_energy = []
        for item in unique_pdgs:
            unique_x.append(x[(pdg == item)])
            unique_y.append(y[(pdg == item)])
            unique_z.append(z[(pdg == item)])
            unique_energy.append(energy[(pdg == item)])
        if plot_type == '3d':
            fig = plt.figure(figsize=(8,6))
            axs = fig.add_subplot(projection='3d')
            for ii in range(len(unique_pdgs)):
                axs.scatter(
                    unique_x[ii], 
                    unique_z[ii], 
                    unique_y[ii],
                    label=f'{pdg_map[str(unique_pdgs[ii])]}',
                    s=20 * unique_energy[ii]
                )
            axs.set_xlabel("x (mm)")
            axs.set_ylabel("z (mm)")
            axs.set_zlabel("y (mm)")
            # draw the active tpc volume box
        else:
            fig, axs = plt.subplots(figsize=(8,6))
            if plot_type == 'xz':
                for ii in range(len(unique_pdgs)):
                    axs.scatter(
                        unique_x[ii], 
                        unique_z[ii], 
                        label=f'{pdg_map[str(unique_pdgs[ii])]}',
                        s=20 * unique_energy[ii]
                    )
                axs.set_xlabel("x (mm)")
                axs.set_ylabel("z (mm)")
            elif plot_type == 'yz':
                for ii in range(len(unique_pdgs)):
                    axs.scatter(
                        unique_y[ii],
                        unique_z[ii],
                        label=f'{pdg_map[str(unique_pdgs[ii])]}',
                        s=20 * unique_energy[ii]
                    )
                axs.set_xlabel("y (mm)")
                axs.set_ylabel("z (mm)")
            else:
                for ii in range(len(unique_pdgs)):
                    axs.scatter(
                        unique_x[ii], 
                        unique_y[ii],
                        label=f'{pdg_map[str(unique_pdgs[ii])]}',
                        s=20 * unique_energy[ii]
                    )
                axs.set_xlabel("x (mm)")
                axs.set_ylabel("y (mm)")
        if show_active_tpc:
            for i in range(len(self.active_tpc_lines)):
                x = np.array([
                    self.active_tpc_lines[i][0][0],
                    self.active_tpc_lines[i][1][0]
                ])
                y = np.array([
                    self.active_tpc_lines[i][0][1],
                    self.active_tpc_lines[i][1][1]
                ])
                z = np.array([
                    self.active_tpc_lines[i][0][2],
                    self.active_tpc_lines[i][1][2]
                ])
                if plot_type == '3d':
                    if i == 0:
                        axs.plot(
                            x,z,y,
                            linestyle='--',color='b',
                            label='Active TPC volume'
                        )
                    else:
                        axs.plot(x,z,y,linestyle='--',color='b')
                elif plot_type == 'xz':
                    if i == 0:
                        axs.plot(
                            x,z,
                            linestyle='--',color='b',
                            label='Active TPC volume'
                        )
                    else:
                        axs.plot(x,z,linestyle='--',color='b')
                elif plot_type == 'yz':
                    if i == 0:
                        axs.plot(
                            y,z,
                            linestyle='--',color='b',
                            label='Active TPC volume'
                        )
                    else:
                        axs.plot(y,z,linestyle='--',color='b')
                else:
                    if i == 0:
                        axs.plot(
                            x,y,
                            linestyle='--',color='b',
                            label='Active TPC volume'
                        )
                    else:
                        axs.plot(x,y,linestyle='--',color='b')
        if show_cryostat:
            for i in range(len(self.cryostat_lines)):
                x = np.array([
                    self.cryostat_lines[i][0][0],
                    self.cryostat_lines[i][1][0]
                ])
                y = np.array([
                    self.cryostat_lines[i][0][1],
                    self.cryostat_lines[i][1][1]
                ])
                z = np.array([
                    self.cryostat_lines[i][0][2],
                    self.cryostat_lines[i][1][2]
                ])
                if plot_type == '3d':
                    if i == 0:
                        axs.plot(
                            x,z,y,
                            linestyle=':',color='g',
                            label='Cryostat volume'
                        )
                    else:
                        axs.plot(x,z,y,linestyle=':',color='g')
                elif plot_type == 'xz':
                    if i == 0:
                        axs.plot(
                            x,z,
                            linestyle=':',color='g',
                            label='Cryostat volume'
                        )
                    else:
                        axs.plot(x,z,linestyle=':',color='g')
                elif plot_type == 'yz':
                    if i == 0:
                        axs.plot(
                            y,z,
                            linestyle=':',color='g',
                            label='Cryostat volume'
                        )
                    else:
                        axs.plot(y,z,linestyle=':',color='g')
                else:
                    if i == 0:
                        axs.plot(
                            x,y,
                            linestyle=':',color='g',
                            label='Cryostat volume'
                        )
                    else:
                        axs.plot(x,y,linestyle=':',color='g')
        axs.set_title(title)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f'plots/{self.name}/events/mc_edep_{plot_type}_{event}.png')
        if show:
            plt.show()

    def plot_mc_voxel_locations(self,
        event,
        plot_type:      str='3d',
        capture_location:str='tpc',
        show_active_tpc:bool=True,
        show_cryostat:  bool=True,
        title:          str='Example MC Voxel Locations',
        save:           bool=True,
        show:           bool=False,
    ):
        if self.load_mc_voxels == False:
            self.logger.error(f"Dataset does not have 'mc_energy_deposits' products loaded! (i.e. 'self.load_mc_voxels' = {self.load_mc_voxels})")
            raise ValueError(f"Dataset does not have 'mc_energy_deposits' products loaded! (i.e. 'self.load_mc_voxels' = {self.load_mc_voxels})")
        if event >= self.num_mc_voxel_events:
            self.logger.error(f"Tried accessing element {event} of array with size {self.num_mc_voxel_events}!")
            raise IndexError(f"Tried accessing element {event} of array with size {self.num_mc_voxel_events}!")
        if plot_type not in ['3d', 'xy', 'xz', 'yz']:
            self.logger.warning(f"Requested plot type '{plot_type}' not allowed, using '3d'.")
            plot_type = '3d'
        if capture_location not in ['world', 'cryostat', 'tpc']:
            self.logger.warning(f"Requested capture location '{capture_location}' not allowed, using 'tpc'.")
            capture_location = 'tpc'
        # gather x, y, z values
        voxels = self.mc_voxels['voxels'][event]
        x = np.array([voxel[0] for voxel in voxels])
        y = np.array([voxel[1] for voxel in voxels])
        z = np.array([voxel[2] for voxel in voxels])
        labels = np.array(self.mc_voxels['labels'][event])
        energy = self.mc_voxels['energy'][event]
        edep_idxs = self.mc_voxels['edep_idxs'][event]
        unique_labels = np.unique(labels)
        unique_x = []
        unique_y = []
        unique_z = []
        unique_energy = []
        for item in unique_labels:
            unique_x.append(x[(labels == item)])
            unique_y.append(y[(labels == item)])
            unique_z.append(z[(labels == item)])
            unique_energy.append(energy[(labels == item)])
        if plot_type == '3d':
            fig = plt.figure(figsize=(8,6))
            axs = fig.add_subplot(projection='3d')
            for ii in range(len(unique_labels)):
                axs.scatter(
                    unique_x[ii], 
                    unique_z[ii], 
                    unique_y[ii],
                    label=f'{unique_labels[ii]}',
                    #s=20 * unique_energy[ii]
                )
            axs.set_xlabel("x (mm)")
            axs.set_ylabel("z (mm)")
            axs.set_zlabel("y (mm)")
            # draw the active tpc volume box
        else:
            fig, axs = plt.subplots(figsize=(8,6))
            if plot_type == 'xz':
                for ii in range(len(unique_labels)):
                    axs.scatter(
                        unique_x[ii], 
                        unique_z[ii], 
                        label=f'{unique_labels[ii]}',
                        #s=20 * unique_energy[ii]
                    )
                axs.set_xlabel("x (mm)")
                axs.set_ylabel("z (mm)")
            elif plot_type == 'yz':
                for ii in range(len(unique_labels)):
                    axs.scatter(
                        unique_y[ii],
                        unique_z[ii],
                        label=f'{unique_labels[ii]}',
                        #s=20 * unique_energy[ii]
                    )
                axs.set_xlabel("y (mm)")
                axs.set_ylabel("z (mm)")
            else:
                for ii in range(len(unique_labels)):
                    axs.scatter(
                        unique_x[ii], 
                        unique_y[ii],
                        label=f'{unique_labels[ii]}',
                        #s=20 * unique_energy[ii]
                    )
                axs.set_xlabel("x (mm)")
                axs.set_ylabel("y (mm)")
        # if show_active_tpc:
        #     for i in range(len(self.active_tpc_lines)):
        #         x = np.array([
        #             self.active_tpc_lines[i][0][0],
        #             self.active_tpc_lines[i][1][0]
        #         ])
        #         y = np.array([
        #             self.active_tpc_lines[i][0][1],
        #             self.active_tpc_lines[i][1][1]
        #         ])
        #         z = np.array([
        #             self.active_tpc_lines[i][0][2],
        #             self.active_tpc_lines[i][1][2]
        #         ])
        #         if plot_type == '3d':
        #             if i == 0:
        #                 axs.plot(
        #                     x,z,y,
        #                     linestyle='--',color='b',
        #                     label='Active TPC volume'
        #                 )
        #             else:
        #                 axs.plot(x,z,y,linestyle='--',color='b')
        #         elif plot_type == 'xz':
        #             if i == 0:
        #                 axs.plot(
        #                     x,z,
        #                     linestyle='--',color='b',
        #                     label='Active TPC volume'
        #                 )
        #             else:
        #                 axs.plot(x,z,linestyle='--',color='b')
        #         elif plot_type == 'yz':
        #             if i == 0:
        #                 axs.plot(
        #                     y,z,
        #                     linestyle='--',color='b',
        #                     label='Active TPC volume'
        #                 )
        #             else:
        #                 axs.plot(y,z,linestyle='--',color='b')
        #         else:
        #             if i == 0:
        #                 axs.plot(
        #                     x,y,
        #                     linestyle='--',color='b',
        #                     label='Active TPC volume'
        #                 )
        #             else:
        #                 axs.plot(x,y,linestyle='--',color='b')
        # if show_cryostat:
        #     for i in range(len(self.cryostat_lines)):
        #         x = np.array([
        #             self.cryostat_lines[i][0][0],
        #             self.cryostat_lines[i][1][0]
        #         ])
        #         y = np.array([
        #             self.cryostat_lines[i][0][1],
        #             self.cryostat_lines[i][1][1]
        #         ])
        #         z = np.array([
        #             self.cryostat_lines[i][0][2],
        #             self.cryostat_lines[i][1][2]
        #         ])
        #         if plot_type == '3d':
        #             if i == 0:
        #                 axs.plot(
        #                     x,z,y,
        #                     linestyle=':',color='g',
        #                     label='Cryostat volume'
        #                 )
        #             else:
        #                 axs.plot(x,z,y,linestyle=':',color='g')
        #         elif plot_type == 'xz':
        #             if i == 0:
        #                 axs.plot(
        #                     x,z,
        #                     linestyle=':',color='g',
        #                     label='Cryostat volume'
        #                 )
        #             else:
        #                 axs.plot(x,z,linestyle=':',color='g')
        #         elif plot_type == 'yz':
        #             if i == 0:
        #                 axs.plot(
        #                     y,z,
        #                     linestyle=':',color='g',
        #                     label='Cryostat volume'
        #                 )
        #             else:
        #                 axs.plot(y,z,linestyle=':',color='g')
        #         else:
        #             if i == 0:
        #                 axs.plot(
        #                     x,y,
        #                     linestyle=':',color='g',
        #                     label='Cryostat volume'
        #                 )
        #             else:
        #                 axs.plot(x,y,linestyle=':',color='g')
        axs.set_title(title)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f'plots/{self.name}/events/mc_voxels_{plot_type}_{event}.png')
        if show:
            plt.show()

    def fit_depth_exponential(self,
        num_bins:   int=100,
        save:       bool=True,
        show:       bool=False
    ):
        y_pos = np.concatenate(
            [yi for yi in self.neutron['neutron_capture_y']]
        ).flatten()
        mask = ((y_pos < self.tpc_y[1]) & (y_pos > self.tpc_y[0]))
        y_pos = y_pos[mask]
        # normalize positions
        y_max = np.max(y_pos)
        depth = np.abs(y_max - y_pos)
        # fit histogram
        y_hist, y_edges = np.histogram(depth, bins=num_bins)
        hist_sum = sum(y_hist)
        y_hist = y_hist.astype(float) / hist_sum
        # determine cumulative hist
        cum_hist = [y_hist[0]]
        for ii in range(1,len(y_hist)):
            cum_hist.append(y_hist[ii]+cum_hist[-1])
        # arrange mid points for fit
        mid_points = np.array([
            y_edges[ii] + (y_edges[ii+1]-y_edges[ii])/2. 
            for ii in range(len(y_hist))
        ])
        # fit to logarithm
        exp_fit = sp.optimize.curve_fit(
            lambda t,a,b: a*np.exp(-b*t),   # decaying exponential
            mid_points, 
            y_hist
        )
        exp_function = exp_fit[0][0] * np.exp(-exp_fit[0][1] * mid_points)
        # plot the results
        fig, axs1 = plt.subplots(figsize=(8,6))
        axs1.scatter(mid_points, y_hist, label='hist')
        axs1.plot(
            mid_points, 
            exp_function, 
            label=rf'fit ($\sim\exp[-{round(exp_fit[0][1],3)}\, \Delta y]$)'
        )
        axs1.set_xlabel(r'depth - $\Delta y$ - (mm)')
        axs1.set_ylabel('density (height/sum)')
        plt.legend(loc='center right')
        axs2 = axs1.twinx()
        axs2.plot(
            mid_points,
            cum_hist,
        )
        axs2.set_ylabel('cummulative %')
        axs1.set_title('Capture density vs. depth (mm)')
        plt.grid(True)
        
        plt.tight_layout()
        if save:
            plt.savefig(f'plots/{self.name}/depth_exponential.png')
        if show:
            plt.show()
    
    def plot_capture_density(self,
        plot_type:      str='xy',
        density_type:   str='kde',
        capture_location:str='tpc',
        title:  str='Example MC Capture Locations',
        save:   bool=True,
        show:   bool=False,
    ):
        if self.load_neutrons == False:
            self.logger.error(f"Dataset does not have 'neutron' products loaded! (i.e. 'self.load_neutrons' = {self.load_neutrons})")
            raise ValueError(f"Dataset does not have 'neutron' products loaded! (i.e. 'self.load_neutrons' = {self.load_neutrons})")
        if plot_type not in ['xy', 'xz', 'yz']:
            self.logger.warning(f"Requested plot type '{plot_type}' not allowed, using 'xy'.")
            plot_type = 'xy'
        if density_type not in ['scatter', 'kde', 'hist', 'hex', 'reg', 'resid']:
            self.logger.warning(f"Requested density type {density_type} not allowed, using 'kde'.")
            density_type = 'kde'
        if capture_location not in ['world', 'cryostat', 'tpc']:
            self.logger.warning(f"Requested capture location '{capture_location}' not allowed, using 'tpc'.")
            capture_location = 'tpc'
        # gather x, y, z values
        x = self.neutron['neutron_capture_x']
        y = self.neutron['neutron_capture_y']
        z = self.neutron['neutron_capture_z']
        x = np.concatenate([
            xi for xi in x
        ]).flatten()
        y = np.concatenate([
            yi for yi in y
        ]).flatten()
        z = np.concatenate([
            zi for zi in z
        ]).flatten()
        if capture_location == 'world':
            mask = (
                (x < self.world_x[1]) & (x > self.world_x[0]) &
                (y < self.world_y[1]) & (y > self.world_y[0]) &
                (z < self.world_z[1]) & (z > self.world_z[0])
            )
        elif capture_location == 'cryostat':
            mask = (
                (x < self.cryo_x[1]) & (x > self.cryo_x[0]) &
                (y < self.cryo_y[1]) & (y > self.cryo_y[0]) &
                (z < self.cryo_z[1]) & (z > self.cryo_z[0])
            )
        else:
            mask = (
                (x < self.tpc_x[1]) & (x > self.tpc_x[0]) &
                (y < self.tpc_y[1]) & (y > self.tpc_y[0]) &
                (z < self.tpc_z[1]) & (z > self.tpc_z[0])
            )
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if plot_type == 'xz':
            sns.jointplot(x=x, y=z, kind=density_type, palette='crest')
            plt.xlabel("x (mm)")
            plt.ylabel("z (mm)")
        elif plot_type == 'yz':
            sns.jointplot(x=y, y=z, kind=density_type, palette='crest')
            plt.xlabel("y (mm)")
            plt.ylabel("z (mm)")
        else:
            sns.jointplot(x=x, y=y, kind=density_type, palette='crest')
            plt.xlabel("x (mm)")
            plt.ylabel("y (mm)")
        plt.title(title)
        plt.tight_layout()
        if save != '':
            plt.savefig(f'plots/{self.name}/capture_density.png')
        if show:
            plt.show()

    def generate_unet_training(self,
        output_file:    str
    ):
        self.logger.info(f"Attempting to generate voxel dataset {output_file}.")
        np.savez(output_file,
            coords= self.mc_voxels['voxels'],
            feats = self.discrete_voxel_values,
            labels= self.mc_voxels['labels'],
            energy= self.mc_voxels['energy'],
            edep_idxs= self.mc_voxels['edep_idxs'],
        )
        self.logger.info(f"Saved voxel dataset to {output_file}.")
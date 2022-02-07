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
        input_file,
        load_neutrons:  bool=True,
        load_muons:     bool=True,
        load_ar39:      bool=True,
        load_voxels:    bool=True,
    ):
        self.load_neutrons = load_neutrons
        self.load_muons    = load_muons
        self.load_ar39     = load_ar39
        self.load_voxels   = load_voxels
        self.logger = UNetLogger('neutron_dataset', file_mode='w')
        self.logger.info(f"Attempting to load file {input_file}.")
        # load the file
        try:
            self.input_file = uproot.open(input_file)
            self.logger.info(f"Successfully loaded file {input_file}.")
        except Exception:
            self.logger.error(f"Failed to load file with exception: {Exception}.")
            raise Exception
        if not os.path.isdir("plots/"):
            os.mkdir("plots/")
        # now load the various arrays
        self.meta       = self.load_array(self.input_file, 'ana/meta')
        self.geometry   = self.load_array(self.input_file, 'ana/Geometry')
        if load_neutrons:
            self.neutron    = self.load_array(self.input_file, 'ana/neutron')
        if load_muons:
            self.muon       = self.load_array(self.input_file, 'ana/muon')
        if load_voxels:
            self.voxels     = self.load_array(self.input_file, 'ana/voxels')

        # construct truth info
        # each index in these arrays correspond to an event
        if self.load_neutrons:
            try:
                self.event_ids          = self.neutron['event_id']
                self.neutron_ids        = self.neutron['neutron_ids']
                self.neutron_capture_x  = self.neutron['neutron_capture_x']
                self.neutron_capture_y  = self.neutron['neutron_capture_y']
                self.neutron_capture_z  = self.neutron['neutron_capture_z']
                self.gamma_ids          = self.neutron['gamma_ids']
                self.gamma_neutron_ids  = self.neutron['gamma_neutron_ids']
                self.gamma_energy       = self.neutron['gamma_energy']
                self.edep_energy        = self.neutron['edep_energy']
                self.edep_parent        = self.neutron['edep_parent']
                self.edep_neutron_ids   = self.neutron['edep_neutron_ids']
                self.edep_gamma_ids     = self.neutron['edep_gamma_ids'] 
                self.neutron_x          = self.neutron['edep_x']
                self.neutron_y          = self.neutron['edep_y']
                self.neutron_z          = self.neutron['edep_z']      
                self.electron_ids           = self.neutron['electron_ids']
                self.electron_neutron_ids   = self.neutron['electron_neutron_ids']
                self.electron_gamma_ids     = self.neutron['electron_gamma_ids']
                self.electron_energy        = self.neutron['electron_energy']
                self.edep_num_electrons     = self.neutron['edep_num_electrons']
            except:
                self.logger.error(f"One or more of the required arrays {required_neutron_arrays} is not present in {self.neutron.keys()}.")
                raise ValueError(f"One or more of the required arrays {required_neutron_arrays} is not present in {self.neutron.keys()}.")
        if self.load_muons:
            try: 
                self.num_muons          = self.muon['primary_muons']
                self.muon_ids           = self.muon['muon_edep_ids']
                self.muon_edep_energy   = self.muon['muon_edep_energy']
                self.muon_edep_num_electrons = self.muon['muon_edep_num_electrons']
                self.muon_edep_x        = self.muon['muon_edep_x']
                self.muon_edep_y        = self.muon['muon_edep_y']
                self.muon_edep_z        = self.muon['muon_edep_z'] 
            except:
                self.logger.error(f"One or more of the required arrays {required_muon_arrays} is not present in {self.muon.keys()}.")
                raise ValueError(f"One or more of the required arrays {required_muon_arrays} is not present in {self.muon.keys()}.")
        if self.load_voxels:
            try:
                self.x_min  = self.voxels['x_min']
                self.x_max  = self.voxels['x_max']
                self.y_min  = self.voxels['y_min']
                self.y_max  = self.voxels['y_max']
                self.z_min  = self.voxels['z_min']
                self.z_max  = self.voxels['z_max']
                self.voxel_size = self.voxels['voxel_size']
                self.num_voxels_x   = self.voxels['num_voxels_x']
                self.num_voxels_y   = self.voxels['num_voxels_y']
                self.num_voxels_z   = self.voxels['num_voxels_z']
                self.x_id   = self.voxels['x_id']
                self.y_id   = self.voxels['y_id']
                self.z_id   = self.voxels['z_id']
                self.voxel_values = self.voxels['values']
                self.voxel_labels = self.voxels['labels']
            except:
                self.logger.error(f"One or more of the required arrays {required_voxel_arrays} is not present in {self.voxels.keys()}.")
                raise ValueError(f"One or more of the required arrays {required_voxel_arrays} is not present in {self.voxels.keys()}.")
        self.num_events = len(self.event_ids)
        self.logger.info(f"Loaded arrays with {self.num_events} entries.")
        if self.load_neutrons:
            # construct positions for neutrons
            self.neutron_edep_positions = np.array(
                [
                    np.array([[
                        self.neutron_x[jj][ii],
                        self.neutron_y[jj][ii],
                        self.neutron_z[jj][ii]]
                        for ii in range(len(self.neutron_x[jj]))
                    ], dtype=float)
                    for jj in range(len(self.neutron_x))
                ], 
                dtype=object
            )
        if self.load_muons:
            self.muon_edep_positions = np.array(
                [
                    np.array([[
                        self.muon_edep_x[jj][ii],
                        self.muon_edep_y[jj][ii],
                        self.muon_edep_z[jj][ii]]
                        for ii in range(len(self.muon_edep_x[jj]))
                    ], dtype=float)
                    for jj in range(len(self.muon_edep_x))
                ], 
                dtype=object
            )
        if self.load_voxels:
            self.voxel_coords = np.array([
                [
                    [
                        self.x_id[j][i],
                        self.y_id[j][i],
                        self.z_id[j][i]]
                        for i in range(len(self.x_id[j]))
                    ]
                    for j in range(len(self.x_id))
                ], 
                dtype=object
            )
            self.discrete_voxel_values = [
                [[1.] for i in range(len(self.voxel_values[i]))] 
                    for i in range(len(self.voxel_values))
                ]
        # construct TPC boxes
        self.total_tpc_ranges = self.geometry['total_active_tpc_box_ranges']
        self.tpc_x = [self.total_tpc_ranges[0][0], self.total_tpc_ranges[0][1]]
        self.tpc_y = [self.total_tpc_ranges[0][4], self.total_tpc_ranges[0][5]]
        self.tpc_z = [self.total_tpc_ranges[0][2], self.total_tpc_ranges[0][3]]
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
        self.cryo_y = [self.total_cryo_ranges[0][4], self.total_cryo_ranges[0][5]]
        self.cryo_z = [self.total_cryo_ranges[0][2], self.total_cryo_ranges[0][3]]
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

    def calculate_capture_ratio(self):
        # keep track of the ratio of complete 6.098
        # captures vs total.
        self.logger.info(f"Attempting to calculate capture ratio.")
        self.num_complete_captures = []
        self.num_captures = []
        # loop through all edeps
        for ii, truth in enumerate(self.edep_neutron_ids):
            complete = 0
            total = 0
            clusters = np.unique(truth)
            for cluster in clusters:
                indices = np.where(truth==cluster)
                true_energies = sum(self.edep_energy[ii][indices])
                total += 1
                if round(true_energies,2) == 6.1:
                    complete += 1
            self.num_captures.append(total)
            self.num_complete_captures.append(complete)
        self.capture_ratio = round((sum(self.num_complete_captures)/sum(self.num_captures))*100)

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

    def plot_event(self,
        event,
        title:  str='',
        show_active_tpc: bool=True,
        show_cryostat:   bool=True,
        save:   str='',
        show:   bool=True,
    ):
        if event >= self.num_events:
            self.logger.error(f"Tried accessing element {event} of array with size {self.num_events}!")
            raise IndexError(f"Tried accessing element {event} of array with size {self.num_events}!")
        fig = plt.figure(figsize=(8,6))
        axs = fig.add_subplot(projection='3d')
        if self.load_neutrons:
            axs.scatter3D(
                self.neutron_x[event], 
                self.neutron_z[event], 
                self.neutron_y[event], 
                label='neutrons', 
            )
        if self.load_muons:
            axs.scatter3D(
                self.muon_edep_x[event],
                self.muon_edep_z[event], 
                self.muon_edep_y[event], 
                label='cosmics', 
            )
        axs.set_xlabel("x (mm)")
        axs.set_ylabel("z (mm)")
        axs.set_zlabel("y (mm)")
        axs.set_title(title)
        # draw the active tpc volume box
        if show_active_tpc:
            for i in range(len(self.active_tpc_lines)):
                x = np.array([self.active_tpc_lines[i][0][0],self.active_tpc_lines[i][1][0]])
                y = np.array([self.active_tpc_lines[i][0][1],self.active_tpc_lines[i][1][1]])
                z = np.array([self.active_tpc_lines[i][0][2],self.active_tpc_lines[i][1][2]])
                if i == 0:
                    axs.plot(x,y,z,linestyle='--',color='b',label='Active TPC volume')
                else:
                    axs.plot(x,y,z,linestyle='--',color='b')
        if show_cryostat:
            for i in range(len(self.cryostat_lines)):
                x = np.array([self.cryostat_lines[i][0][0],self.cryostat_lines[i][1][0]])
                y = np.array([self.cryostat_lines[i][0][1],self.cryostat_lines[i][1][1]])
                z = np.array([self.cryostat_lines[i][0][2],self.cryostat_lines[i][1][2]])
                if i == 0:
                    axs.plot(x,y,z,linestyle=':',color='g',label='Cryostat volume')
                else:
                    axs.plot(x,y,z,linestyle=':',color='g')
        plt.legend()
        plt.tight_layout()
        if save != '':
            plt.savefig('plots/'+save+'.png')
        if show:
            plt.show()
    
    def plot_event_neutrons(self,
        event,
        label:  str='neutron',  # plot by neutron, gamma, electron
        title:  str='',
        legend_cutoff:  int=10, # only show the first N labels in the legend (gets crammed easily)
        show_active_tpc: bool=True,
        show_cryostat:   bool=True,
        save:   str='',
        show:   bool=True,
    ):
        if event >= self.num_events:
            self.logger.error(f"Tried accessing element {event} of array with size {self.num_events}!")
            raise IndexError(f"Tried accessing element {event} of array with size {self.num_events}!")
        if label not in ['neutron', 'gamma']:
            self.logger.warning(f"Requested labeling by '{label}' not allowed, using 'neutron'.")
            label = 'neutron'
        fig = plt.figure(figsize=(8,6))
        axs = fig.add_subplot(projection='3d')
        x, y, z, ids, energy = [], [], [], [], []
        indices = []
        if label == 'neutron':
            labels = np.unique(self.edep_neutron_ids[event])
            for ii, value in enumerate(labels):
                indices.append(np.where(self.edep_neutron_ids[event] == value))
        else:
            labels = np.unique(self.edep_gamma_ids[event])
            for ii, value in enumerate(labels):
                indices.append(np.where(self.edep_gamma_ids[event] == value))
        for ii, value in enumerate(labels):
            x.append([self.neutron_x[event][indices[ii]]])
            y.append([self.neutron_z[event][indices[ii]]])
            z.append([self.neutron_y[event][indices[ii]]])
            energy.append([1000*self.edep_energy[event][indices[ii]]])
            ids.append(f'{label} {ii}: (id: {value}, energy: {round(sum(self.edep_energy[event][indices[ii]]),4)} MeV)')
        for jj in range(len(x)):
            if jj < legend_cutoff:
                axs.scatter3D(x[jj], y[jj], z[jj], label=ids[jj], s=energy[jj])
            else:
                axs.scatter3D(x[jj], y[jj], z[jj], s=energy[jj])
        axs.set_xlabel("x (mm)")
        axs.set_ylabel("z (mm)")
        axs.set_zlabel("y (mm)")
        axs.set_title(title)
        # draw the active tpc volume box
        if show_active_tpc:
            for i in range(len(self.active_tpc_lines)):
                x = np.array([self.active_tpc_lines[i][0][0],self.active_tpc_lines[i][1][0]])
                y = np.array([self.active_tpc_lines[i][0][1],self.active_tpc_lines[i][1][1]])
                z = np.array([self.active_tpc_lines[i][0][2],self.active_tpc_lines[i][1][2]])
                if i == 0:
                    axs.plot(x,y,z,linestyle='--',color='b',label='Active TPC volume')
                else:
                    axs.plot(x,y,z,linestyle='--',color='b')
        if show_cryostat:
            for i in range(len(self.cryostat_lines)):
                x = np.array([self.cryostat_lines[i][0][0],self.cryostat_lines[i][1][0]])
                y = np.array([self.cryostat_lines[i][0][1],self.cryostat_lines[i][1][1]])
                z = np.array([self.cryostat_lines[i][0][2],self.cryostat_lines[i][1][2]])
                if i == 0:
                    axs.plot(x,y,z,linestyle=':',color='g',label='Cryostat volume')
                else:
                    axs.plot(x,y,z,linestyle=':',color='g')
        plt.legend()
        plt.tight_layout()
        if save != '':
            plt.savefig('plots/'+save+'.png')
        if show:
            plt.show()

    def fit_depth_exponential(self,
        num_bins:   int=100,
        save:   str='',
        show:   bool=True
    ):
        y_pos = np.concatenate([yi for yi in self.neutron_y]).flatten()
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
        mid_points = np.array([y_edges[ii] + (y_edges[ii+1]-y_edges[ii])/2. for ii in range(len(y_hist))])
        # fit to logarithm
        exp_fit = sp.optimize.curve_fit(
            lambda t,a,b: a*np.exp(-b*t),   # decaying exponential
            mid_points, 
            y_hist
        )
        exp_function = exp_fit[0][0] * np.exp(-exp_fit[0][1] * mid_points)
        # plot the results
        fig, axs = plt.subplots(figsize=(8,6))
        axs.scatter(mid_points, y_hist, label='hist')
        axs.plot(
            mid_points, 
            exp_function, 
            label=rf'fit ($\sim\exp[-{round(exp_fit[0][1],3)}\, \Delta y]$)'
        )
        axs.set_xlabel(r'depth - $\Delta y$ - (mm)')
        axs.set_ylabel('density (height/sum)')
        plt.legend(loc='center right')
        axs2 = axs.twinx()
        axs2.plot(
            mid_points,
            cum_hist,
        )
        axs2.set_ylabel('cummulative %')
        axs.set_title('Capture density vs. depth (mm)')
        plt.grid(True)
        
        plt.tight_layout()
        if save != '':
            plt.savefig('plots/'+save+'.png')
        if show:
            plt.show()
    
    def plot_capture_locations(self,
        event:          int,
        plot_type:      str='3d',
        show_active_tpc:bool=True,
        show_cryostat:  bool=True,
        title:  str='Example MC Capture Locations',
        save:   str='',
        show:   bool=True,
    ):
        if event >= self.num_events:
            self.logger.error(f"Tried accessing element {event} of array with size {self.num_events}!")
            raise IndexError(f"Tried accessing element {event} of array with size {self.num_events}!")
        if plot_type not in ['3d', 'xy', 'xz', 'yz']:
            self.logger.warning(f"Requested plot type '{plot_type}' not allowed, using '3d'.")
            plot_type = '3d'
        if plot_type == '3d':
            fig = plt.figure(figsize=(8,6))
            axs = fig.add_subplot(projection='3d')
            axs.scatter(
                self.neutron_capture_x[event],
                self.neutron_capture_z[event],
                self.neutron_capture_y[event]
            )
            axs.set_xlabel("x (mm)")
            axs.set_ylabel("z (mm)")
            axs.set_zlabel("y (mm)")
            # draw the active tpc volume box
        else:
            fig, axs = plt.subplots(figsize=(8,6))
            if plot_type == 'xz':
                axs.scatter(self.neutron_capture_x[event], self.neutron_capture_z[event])
                axs.set_xlabel("x (mm)")
                axs.set_ylabel("z (mm)")
            elif plot_type == 'yz':
                axs.scatter(self.neutron_capture_y[event], self.neutron_capture_z[event])
                axs.set_xlabel("y (mm)")
                axs.set_ylabel("z (mm)")
            else:
                axs.scatter(self.neutron_capture_x[event], self.neutron_capture_y[event])
                axs.set_xlabel("x (mm)")
                axs.set_ylabel("y (mm)")
        if show_active_tpc:
            for i in range(len(self.active_tpc_lines)):
                x = np.array([self.active_tpc_lines[i][0][0],self.active_tpc_lines[i][1][0]])
                y = np.array([self.active_tpc_lines[i][0][1],self.active_tpc_lines[i][1][1]])
                z = np.array([self.active_tpc_lines[i][0][2],self.active_tpc_lines[i][1][2]])
                if plot_type == '3d':
                    if i == 0:
                        axs.plot(x,y,z,linestyle='--',color='b',label='Active TPC volume')
                    else:
                        axs.plot(x,y,z,linestyle='--',color='b')
                elif plot_type == 'xz':
                    if i == 0:
                        axs.plot(x,y,linestyle='--',color='b',label='Active TPC volume')
                    else:
                        axs.plot(x,y,linestyle='--',color='b')
                elif plot_type == 'yz':
                    if i == 0:
                        axs.plot(z,y,linestyle='--',color='b',label='Active TPC volume')
                    else:
                        axs.plot(z,y,linestyle='--',color='b')
                else:
                    if i == 0:
                        axs.plot(x,z,linestyle='--',color='b',label='Active TPC volume')
                    else:
                        axs.plot(x,z,linestyle='--',color='b')
        if show_cryostat:
            for i in range(len(self.cryostat_lines)):
                x = np.array([self.cryostat_lines[i][0][0],self.cryostat_lines[i][1][0]])
                y = np.array([self.cryostat_lines[i][0][1],self.cryostat_lines[i][1][1]])
                z = np.array([self.cryostat_lines[i][0][2],self.cryostat_lines[i][1][2]])
                if plot_type == '3d':
                    if i == 0:
                        axs.plot(x,y,z,linestyle=':',color='g',label='Cryostat volume')
                    else:
                        axs.plot(x,y,z,linestyle=':',color='g')
                elif plot_type == 'xz':
                    if i == 0:
                        axs.plot(x,y,linestyle=':',color='g',label='Cryostat volume')
                    else:
                        axs.plot(x,y,linestyle=':',color='g')
                elif plot_type == 'yz':
                    if i == 0:
                        axs.plot(z,y,linestyle=':',color='g',label='Cryostat volume')
                    else:
                        axs.plot(z,y,linestyle=':',color='g')
                else:
                    if i == 0:
                        axs.plot(x,z,linestyle=':',color='g',label='Cryostat volume')
                    else:
                        axs.plot(x,z,linestyle=':',color='g')
        axs.set_title(title)
        plt.legend()
        plt.tight_layout()
        if save != '':
            plt.savefig('plots/'+save+'.png')
        if show:
            plt.show()

    def plot_capture_density(self,
        plot_type:      str='xy',
        density_type:   str='kde',
        title:  str='Example MC Capture Locations',
        save:   str='',
        show:   bool=True,
    ):
        if plot_type not in ['xy', 'xz', 'yz']:
            self.logger.warning(f"Requested plot type '{plot_type}' not allowed, using 'xy'.")
            plot_type = 'xy'
        if density_type not in ['scatter', 'kde', 'hist', 'hex', 'reg', 'resid']:
            self.logger.warning(f"Requested density type {density_type} not allowed, using 'kde'.")
            density_type = 'kde'
        if plot_type == 'xz':
            x = np.concatenate([xi for xi in self.neutron_capture_x]).flatten()
            y = np.concatenate([zi for zi in self.neutron_capture_z]).flatten()
        elif plot_type == 'yz':
            x = np.concatenate([yi for yi in self.neutron_capture_y]).flatten()
            y = np.concatenate([zi for zi in self.neutron_capture_z]).flatten()
        else:
            x = np.concatenate([xi for xi in self.neutron_capture_x]).flatten()
            y = np.concatenate([yi for yi in self.neutron_capture_y]).flatten()
        sns.jointplot(x=x, y=y, kind=density_type, palette='crest')
        if plot_type == 'xz':
            plt.xlabel("x (mm)")
            plt.ylabel("z (mm)")
        elif plot_type == 'yz':
            plt.xlabel("y (mm)")
            plt.ylabel("z (mm)")
        else:
            plt.xlabel("x (mm)")
            plt.ylabel("y (mm)")
        plt.title(title)
        plt.tight_layout()
        if save != '':
            plt.savefig('plots/'+save+'.png')
        if show:
            plt.show()

    def generate_unet_training(self,
        output_file:    str
    ):
        self.logger.info(f"Attempting to generate voxel dataset {output_file}.")
        np.savez(output_file,
            coords= self.voxel_coords,
            feats = self.discrete_voxel_values,
            labels= self.voxel_labels,
            energy= self.voxel_values
        )
        self.logger.info(f"Saved voxel dataset to {output_file}.")
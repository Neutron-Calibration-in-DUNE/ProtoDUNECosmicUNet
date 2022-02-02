"""
Collection of classes for generating cosmic ray datasets
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
import logging
from sklearn import cluster
from sklearn import metrics
import seaborn as sns
import MinkowskiEngine as ME
#plt.style.use('seaborn-deep')

# set up logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("protodune_cosmic.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_array(
    input_file,
    array_name,
):
    logging.info(f"Attempting to load array: {array_name} from file: {input_file}.")
    try:
        array = input_file[array_name].arrays(library="np")
        logging.info(f"Successfully loaded array: {array_name} from file: {input_file}.")
    except Exception:
        logging.error(f"Failed to load array: {array_name} from file: {input_file} with exception: {Exception}.")
        raise Exception
    return array

required_arrays = [
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
    'primary_muons',
    'muon_ids',
    'muon_edep_ids',
    'muon_edep_energy',
    'muon_edep_num_electrons',
    'muon_edep_x',
    'muon_edep_y',
    'muon_edep_z',
    'electron_ids',
    'electron_neutron_ids',
    'electron_gamma_ids',
    'electron_energy',
    'edep_num_electrons',
]

class NeutronCosmicDataset:
    """
    This class loads simulated neutron events and runs various clustering
    algorithms and analysis.  The arrays in the root file should be structured
    as follows:
        meta:       meta information such as ...
        Geometry:   information about the detector geometry such as volume bounding boxes...
        neutron:    the collection of neutron event information from the simulation
        muon:       the collection of muon event information
        
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
        primary_muons:      the number of muons in the event
        muon_ids:           the track ids of the muons
        muon_edep_ids:      the track id of the corresponding muon that left the energy deposit
        muon_edep_energy:   the energy values of each unique deposit
        muon_edep_num_electrons:    the number of electrons generated from each energy deposit
        muon_edep_x:        the x position of each edep from muons
        muon_edep_y:        the y ""
        muon_edep_z:        the z ""
        electron_ids:           the track id of each electron tracked in the simulation that comes from a gamma
        electron_neutron_ids:   the corresponding id of the neutron that generated the electron with id ^^^
        electron_gamma_ids:     the corresponding id of the gamma that generated the electron with id ^^^
        electron_energy (GeV):  the energy of each electron tracked in the simulation (e.g. [0.00058097, ...])
        edep_num_electrons:     the number of electrons coming out of the IonAndScint simulation for each edep ^^^
    """
    def __init__(self,
        input_file,
    ):
        logging.info(f"Attempting to load file {input_file}.")
        # load the file
        try:
            self.input_file = uproot.open(input_file)
            logging.info(f"Successfully loaded file {input_file}.")
        except Exception:
            logging.error(f"Failed to load file with exception: {Exception}.")
            raise Exception
        # now load the various arrays
        self.meta       = load_array(self.input_file, 'ana/meta')
        self.geometry   = load_array(self.input_file, 'ana/Geometry')
        self.events     = load_array(self.input_file, 'ana/neutron')

        # construct truth info
        # each index in these arrays correspond to an event
        try:
            self.event_ids          = self.events['event_id']
            self.neutron_ids        = self.events['neutron_ids']
            self.neutron_capture_x  = self.events['neutron_capture_x']
            self.neutron_capture_y  = self.events['neutron_capture_y']
            self.neutron_capture_z  = self.events['neutron_capture_z']
            self.gamma_ids          = self.events['gamma_ids']
            self.gamma_neutron_ids  = self.events['gamma_neutron_ids']
            self.gamma_energy       = self.events['gamma_energy']
            self.edep_energy        = self.events['edep_energy']
            self.edep_parent        = self.events['edep_parent']
            self.edep_neutron_ids   = self.events['edep_neutron_ids']
            self.edep_gamma_ids     = self.events['edep_gamma_ids'] 
            self.neutron_x          = self.events['edep_x']
            self.neutron_y          = self.events['edep_y']
            self.neutron_z          = self.events['edep_z']
            self.num_muons          = self.events['primary_muons']
            self.muon_ids           = self.events['muon_edep_ids']
            self.muon_edep_energy   = self.events['muon_edep_energy']
            self.muon_edep_num_electrons = self.events['muon_edep_num_electrons']
            self.muon_edep_x        = self.events['muon_edep_x']
            self.muon_edep_y        = self.events['muon_edep_y']
            self.muon_edep_z        = self.events['muon_edep_z']       
            self.electron_ids           = self.events['electron_ids']
            self.electron_neutron_ids   = self.events['electron_neutron_ids']
            self.electron_gamma_ids     = self.events['electron_gamma_ids']
            self.electron_energy        = self.events['electron_energy']
            self.edep_num_electrons     = self.events['edep_num_electrons']
        except:
            logging.error(f"One or more of the required arrays {required_arrays} is not present in {self.events.keys()}.")
            raise ValueError
        self.num_events = len(self.event_ids)
        logging.info(f"Loaded arrays with {self.num_events} entries.")
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

    def plot_event(self,
        index,
        title:  str='',
        show_active_tpc: bool=True,
        show_cryostat:   bool=True,
        save:   str='',
        show:   bool=True,
    ):
        if index >= self.num_events:
            logging.error(f"Tried accessing element {index} of array with size {self.num_events}!")
            raise IndexError(f"Tried accessing element {index} of array with size {self.num_events}!")
        fig = plt.figure(figsize=(8,6))
        axs = fig.add_subplot(projection='3d')
        axs.scatter3D(
            self.neutron_x[index], 
            self.neutron_z[index], 
            self.neutron_y[index], 
            label='neutrons', 
            #s=1000*self.edep_energy[index]
        )
        axs.scatter3D(
            self.muon_edep_x[index],
            self.muon_edep_z[index], 
            self.muon_edep_y[index], 
            label='cosmics', 
            #s=1000*self.muon_edep_energy[index]
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

    def voxelize(self,
        event:  int,
        voxel_size: int=4
    ):
        """
        Generates a voxelized representation of the data.
        """
        num_x_voxels = int(abs(self.tpc_x[1]-self.tpc_x[0])/voxel_size)
        num_y_voxels = int(abs(self.tpc_y[1]-self.tpc_y[0])/voxel_size)
        num_z_voxels = int(abs(self.tpc_z[1]-self.tpc_z[0])/voxel_size)
        voxels = []
        values = []
        labels = []
        for ii in range(num_x_voxels):
            for jj in range(num_y_voxels):
                for kk in range(num_z_voxels):
                    for xyz in range(len(self.neutron_x[event])):
                        if (
                            (self.neutron_x[event][xyz] >= self.tpc_x[0] + ii*voxel_size) and
                            (self.neutron_x[event][xyz] < self.tpc_x[0] + (ii+1)*voxel_size) and 
                            (self.neutron_y[event][xyz] >= self.tpc_y[0] + jj*voxel_size) and
                            (self.neutron_y[event][xyz] < self.tpc_y[0] + (jj+1)*voxel_size) and 
                            (self.neutron_z[event][xyz] >= self.tpc_z[0] + kk*voxel_size) and
                            (self.neutron_z[event][xyz] < self.tpc_z[0] + (kk+1)*voxel_size)
                        ):
                            voxels.append([ii,jj,kk])
                            values.append([self.edep_energy[event][xyz]])
                            labels.append([0])
                    for xyz in range(len(self.muon_edep_x[event])):
                        if (
                            (self.muon_edep_x[event][xyz] >= self.tpc_x[0] + ii*voxel_size) and
                            (self.muon_edep_x[event][xyz] < self.tpc_x[0] + (ii+1)*voxel_size) and 
                            (self.muon_edep_y[event][xyz] >= self.tpc_y[0] + jj*voxel_size) and
                            (self.muon_edep_y[event][xyz] < self.tpc_y[0] + (jj+1)*voxel_size) and 
                            (self.muon_edep_z[event][xyz] >= self.tpc_z[0] + kk*voxel_size) and
                            (self.muon_edep_z[event][xyz] < self.tpc_z[0] + (kk+1)*voxel_size)
                        ):
                            voxels.append([ii,jj,kk])
                            values.append([self.muon_edep_energy[event][xyz]])
                            labels.append([1])
        # consolidate voxels
        logging.info(f"Generated {len(voxels)} voxel values for event {event}.")
        return voxels, values, labels


if __name__ == "__main__":

    dataset = NeutronCosmicDataset(
        input_file="../neutron_data/protodune_cosmic_g4.root"
    )
    print(dataset.neutron_edep_positions[0])
    voxels = ME.utils.quantization.quantize(dataset.neutron_edep_positions[0])

    dataset.plot_event(
        index=0,
        title='ProtoDUNE cosmic example',
        save='protondune_cosmic_g4'
    )


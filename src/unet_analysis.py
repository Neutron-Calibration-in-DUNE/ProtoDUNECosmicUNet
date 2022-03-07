"""
Generic analysis code for the unet results
"""
import numpy as np
import scipy as sp
import uproot
from matplotlib import pyplot as plt
from unet_logger import UNetLogger
from sklearn import cluster
from sklearn import metrics
import csv
import os

cluster_params = {
    'affinity':     {'damping': 0.5, 'max_iter': 200},
    'mean_shift':   {'bandwidth': None},
    'dbscan':       {'eps': 100.,'min_samples': 6},
    'optics':       {'min_samples': 6},
    'gaussian':     {'n_components': 1, 'covariance_type': 'full', 'tol': 1e-3, 'reg_covar': 1e-6, 'max_iter': 100}
}
class UNetAnalyzer:

    def __init__(self,
        name:       str,
        input_file: str,
        source_file:str,
    ):
        self.logger = UNetLogger('analysis', file_mode='w')
        self.name = name
        self.input_file = input_file
        input   = np.load(input_file, allow_pickle=True)
        if not os.path.isdir(f"plots/{self.name}/events/"):
            os.makedirs(f"plots/{self.name}/events/")
        self.events  = input['events']
        self.coords  = input['coords']
        self.feats   = input['feats']
        self.energy  = input['energy']
        self.labels  = input['labels']
        self.preds   = input['predictions']
        self.metrics = input['metrics']
        self.metric_names = input['metric_names']
        self.correct = input['labels']

        for event in range(len(self.coords)):
            if len(np.unique(self.labels[event])) == 2:
                # binary classification
                self.preds[event] = (self.preds[event] > 0.0).astype(int)
                self.correct[event] = (self.preds[event] == self.labels[event])
                self.label_names = ['neutron','cosmic']
        
        source = uproot.open(source_file, allow_pickle=True)
        self.edeps = source['ana/mc_energy_deposits']
        self.edep_idxs = source['ana/mc_voxels']['edep_idxs'].array(library="np")
        self.edep_x = self.edeps['edep_x'].array(library="np")
        self.edep_y = self.edeps['edep_y'].array(library="np")
        self.edep_z = self.edeps['edep_z'].array(library="np")
        self.edep_energy = self.edeps['energy'].array(library="np")
        self.edep_positions = np.array(
            [
                np.array([[
                    self.edep_x[jj][ii],
                    self.edep_y[jj][ii],
                    self.edep_z[jj][ii]]
                    for ii in range(len(self.edep_x[jj]))
                ], dtype=float)
                for jj in range(len(self.edep_x))
            ], 
            dtype=object
        )
        
    def plot_predictions(self,
        event:  int,
        save:   bool=True,
        show:   bool=False,
    ):
        if event > len(self.events):
            self.logger.error(f"Event ID {event} greater than number of events: {len(self.events)}.")
            return
        coords = self.coords[event]
        energy = self.energy[event]
        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]
        feats  = self.feats[event]
        labels = self.labels[event]
        preds  = self.labels[event]

        fig = plt.figure(figsize=(8,6))
        axs_truth = fig.add_subplot(1, 2, 1, projection='3d')
        for label in np.unique(labels):
            temp_x = x[labels == label]
            temp_y = y[labels == label]
            temp_z = z[labels == label]
            axs_truth.scatter3D(
                temp_x,
                temp_z,
                temp_y,
                label=f'true_{self.label_names[int(label)]}'
            )
        axs_truth.legend()
        axs_preds = fig.add_subplot(1, 2, 2, projection='3d')
        for label in np.unique(preds):
            temp_x = x[preds == label]
            temp_y = y[preds == label]
            temp_z = z[preds == label]
            axs_preds.scatter3D(
                temp_x,
                temp_z,
                temp_y,
                label=f'pred_{self.label_names[int(label)]}'
            )
        axs_preds.legend()
        title_str = f'Neutron/Cosmic Semantic Segmentation Results (Event: {event})'
        for ii, metric in enumerate(self.metric_names):
            title_str += f'\n{metric}: {self.metrics[event][ii]}'
        plt.suptitle(title_str)
        plt.tight_layout()
        if save:
            plt.savefig(f'plots/{self.name}/events/pred_{event}.png')
        if show:
            plt.show()

    def cluster_predictions(self,
        level:      str='truth',
        remove_cosmic:  bool=True,
        alg:        str='dbscan',
        params:     dict={'eps': 14./4.7,'min_samples': 6},
        num_bins:   int=100,
        energy_cut: float=10.,   # MeV
        title:      str='Example Prediction Spectrum',
        save:       str='',
        show:       bool=False,
    ):
        """
        Function for running clustering algorithms on events.
        The level can be ['truth','prediction','compare']
        """
        if level not in ['truth', 'prediction','compare']:
            self.logger.warning(f"Requested cluster level by '{level}' not allowed, using 'truth'.")
            level = 'truth'
        if alg not in cluster_params.keys():
            self.logger.warning(f"Requested algorithm '{alg}' not allowed, using 'dbscan'.")
            alg = 'dbscan'
            params = cluster_params['dbscan']
        for item in params:
            if item not in cluster_params[alg]:
                self.logger.error(f"Unrecognized parameter {item} for algorithm {alg}! Available parameters are {cluster_params[alg]}.")
                raise ValueError(f"Unrecognized parameter {item} for algorithm {alg}! Available parameters are {cluster_params[alg]}.")
        # run the clustering algorithm
        self.logger.info(f"Attempting to run clustering algorithm {alg} with parameters {params}.")
        if alg == 'affinity':
            clusterer = cluster.AffinityPropagation(**params)
        elif alg == 'mean_shift':
            clusterer = cluster.MeanShift(**params)
        elif alg == 'optics':
            clusterer = cluster.OPTICS(**params)
        elif alg == 'gaussian':
            clusterer = cluster.GaussianMixture(**params)
        else:
            clusterer = cluster.DBSCAN(**params)
        self.cluster_truth = []
        self.cluster_preds = []
        # collect variables
        for event in range(len(self.events)):
            coords = self.coords[event]
            feats  = self.feats[event]
            labels = self.labels[event]
            preds  = self.labels[event]
            edep_idxs = self.edep_idxs[event]
            # get truth spectrum
            truth_coords = coords[(labels == 0)]
            tmp_edeps = np.array(edep_idxs[(labels == 0)])
            tmp_edeps = [[tmp_edeps[i][j] for j in range(len(tmp_edeps[i]))] for i in range(len(tmp_edeps))]
            tmp_edeps = [item for sublist in tmp_edeps for item in sublist]
            edeps = self.edep_positions[event][tmp_edeps]
            #clusterer.fit(tmp_coords)
            clusterer.fit(edeps)
            #clusterer.fit(truth_coords)
            self.cluster_truth.append(clusterer.labels_)
            # get prediction spectrum
            if remove_cosmic:
                tmp_coords = coords[(preds == 0)]
                tmp_edeps = np.array(edep_idxs[(preds == 0)])
                tmp_edeps = [[tmp_edeps[i][j] for j in range(len(tmp_edeps[i]))] for i in range(len(tmp_edeps))]
                tmp_edeps = [item for sublist in tmp_edeps for item in sublist]
                edeps = self.edep_positions[event][tmp_edeps]
                #clusterer.fit(tmp_coords)
                clusterer.fit(edeps)
                self.cluster_preds.append(clusterer.labels_)
            else:
                cluster.fit(coords)
                self.cluster_preds.append(clusterer.labels_)
    
        truth_spectrum = []
        prediction_spectrum = []
        # construct truth spectrum
        for ii, truth in enumerate(self.cluster_truth):
            clusters = np.unique(truth)
            for c in clusters:
                indices = np.where(truth==c)
                true_energies = sum(self.edep_energy[ii][indices])
                if true_energies < energy_cut:
                    truth_spectrum.append(true_energies)  
        # construct prediction spectrum
        for ii, pred in enumerate(self.cluster_preds):
            clusters = np.unique(pred)
            for c in clusters:
                indices = np.where(pred==c)
                pred_energies = sum(self.edep_energy[ii][indices])
                if pred_energies < energy_cut:
                    prediction_spectrum.append(pred_energies)  
        fig, axs = plt.subplots()
        if level == 'truth' or level == 'compare':
            axs.hist(truth_spectrum, bins=num_bins, label='truth', histtype='step')
        if level == 'prediction' or level == 'compare':
            axs.hist(prediction_spectrum, bins=num_bins, label='pred', histtype='step')
        axs.set_xlabel('Cluster Energy (MeV)')
        axs.set_xticks([1, 2, 3, 4, 5, 6.098, 7, 8])
        axs.set_title(title)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig('plots/{self.name}/cluster_predictions.png')
        if show:
            plt.show()
        
        




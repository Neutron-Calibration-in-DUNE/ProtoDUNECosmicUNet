"""
Collection of clustering functions for neutron calibration sims/data
"""
# imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv
from sklearn import cluster
from sklearn import metrics
import seaborn as sns
import uproot
import os
import sys
sys.path.append("../")
from neutron_dataset import NeutronCosmicDataset
from parameters import *

def euclidean_distance(p1, p2):
    if (len(p1) != len(p2)):
        raise ValueError(f"Tried to compute Euclidean distance between two points with different dimension ({len(p1)},{len(p2)})!")
    dist = [(a - b)**2 for a, b in zip(p1, p2)]
    dist = math.sqrt(sum(dist))
    return dist

def find_largest_distance(pos):
    dists = []
    for p in pos:
        for q in pos:
            dists.append(
                euclidean_distance(p,q)
            )
    return max(dists)

class NeutronClustering(NeutronCosmicDataset):
    """
    This class loads simulated neutron events and runs various clustering
    algorithms and analysis.  The arrays in the root file should be structured
    as follows:
    """
    def __init__(self,
        input_file,
    ):
        super(NeutronClustering, self).__init__(input_file)
        if not os.path.isdir("clustering/"):
            os.mkdir("clustering/")
        self.truth_cluster_predictions = []
        self.truth_cluster_scores = {
            'homogeneity':          [],
            'completeness':         [],
            'v-measure':            [],
            'adjusted_rand_index':  [],
            'adjusted_mutual_info': [],
            'silhouette':           [],
        }
        self.truth_avg_cluster_scores = {
            'homogeneity':          0.,
            'completeness':         0.,
            'v-measure':            0.,
            'adjusted_rand_index':  0.,
            'adjusted_mutual_info': 0.,
            'silhouette':           0.,
        }
        self.cluster_spectrum = []

    # functions involving MC truth clustering
    def cluster_truth(self,
        level:  str='neutron',
        alg:    str='dbscan',
        params: dict={'eps': 100.,'min_samples': 6},
    ):
        """
        Function for running clustering algorithms on events.
        The level can be ['neutron','gamma']
        """
        if level not in ['neutron', 'gamma']:
            self.logger.warning(f"Requested cluster level by '{level}' not allowed, using 'neutron'.")
            level = 'neutron'
        if alg not in cluster_params.keys():
            self.logger.warning(f"Requested algorithm '{alg}' not allowed, using 'dbscan'.")
            alg = 'dbscan'
            params = cluster_params['dbscan']
        # check params
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
        self.truth_cluster_predictions = []
        for pos in self.neutron_edep_positions:
            clusterer.fit(pos)
            self.truth_cluster_predictions.append(clusterer.labels_)

    def calc_truth_scores(self,
        level:  str='neutron',
    ):
        if level not in ['neutron', 'gamma']:
            self.logger.warning(f"Requested cluster level by '{level}' not allowed, using 'neutron'.")
            level = 'neutron'
        if self.truth_cluster_predictions == []:
            self.logger.error("No predictions have been made, need to run clustering algorithm first!")
            raise ValueError("No predictions have been made, need to run clustering algorithm first!")
        if len(self.truth_cluster_predictions) != self.num_events:
            self.logger.error(f"Only {len(self.truth_cluster_predictions)} predictions but {self.num_events} events!")
            raise ValueError(f"Only {len(self.truth_cluster_predictions)} predictions but {self.num_events} events!")
        # clear the scores
        for item in self.truth_cluster_scores.keys():
            self.truth_cluster_scores[item] = []
        if level == 'neutron':
            labels = self.edep_neutron_ids
        else:
            labels = self.edep_gamma_ids
        self.logger.info(f"Attempting to calculate scores on cluster predictions for level: {level}.")
        for ii, pred in enumerate(self.truth_cluster_predictions):
            self.truth_cluster_scores['homogeneity'].append(metrics.homogeneity_score(labels[ii], pred))
            self.truth_cluster_scores['completeness'].append(metrics.completeness_score(labels[ii], pred))
            self.truth_cluster_scores['v-measure'].append(metrics.v_measure_score(labels[ii], pred))
            self.truth_cluster_scores['adjusted_rand_index'].append(metrics.adjusted_rand_score(labels[ii], pred))
            self.truth_cluster_scores['adjusted_mutual_info'].append(metrics.adjusted_mutual_info_score(labels[ii], pred))
            self.truth_cluster_scores['silhouette'].append(metrics.silhouette_score(self.neutron_edep_positions[ii], pred))
        for item in self.truth_cluster_scores.keys():
            self.truth_avg_cluster_scores[item] = sum(self.truth_cluster_scores[item]) / len(labels)
        self.logger.info(f"Calculated average scores {self.truth_avg_cluster_scores} for level: {level}.")
        return self.truth_avg_cluster_scores

    def plot_prediction_spectrum(self,
        num_bins:   int=100,
        edep_type:  str='true',
        energy_cut: float=10.,   # MeV
        title:  str='Example Prediction Spectrum',
        save:   str='',
        show:   bool=True,
    ):
        if edep_type not in ['true','ion_scint','compare']:
            self.logger.warning(f"Requested edep type '{edep_type}' not allowed, using 'true'.")
            edep_type = 'true'
        if self.truth_cluster_predictions == []:
            self.logger.error("No predictions have been made, need to run clustering algorithm first!")
            raise ValueError("No predictions have been made, need to run clustering algorithm first!")
        if len(self.truth_cluster_predictions) != self.num_events:
            self.logger.error(f"Only {len(self.truth_cluster_predictions)} predictions but {self.num_events} events!")
            raise ValueError(f"Only {len(self.truth_cluster_predictions)} predictions but {self.num_events} events!")
        # find cluster energies
        if edep_type == 'compare':
            self.cluster_spectrum = [[],[]]
        else:
            self.cluster_spectrum = [[]]
        for ii, pred in enumerate(self.truth_cluster_predictions):
            clusters = np.unique(pred)
            for cluster in clusters:
                indices = np.where(pred==cluster)
                if edep_type == 'true':
                    true_energies = sum(self.edep_energy[ii][indices])
                    if true_energies < energy_cut:
                        self.cluster_spectrum[0].append(true_energies)
                elif edep_type == 'ion_scint':
                    ion_scint_energies = sum(self.edep_num_electrons[ii][indices]*1.5763e-5)  # eV of Ar ionization (MeV)
                    if ion_scint_energies < energy_cut:
                        self.cluster_spectrum[0].append(ion_scint_energies)
                else:
                    true_energies = sum(self.edep_energy[ii][indices])
                    ion_scint_energies = sum(self.edep_num_electrons[ii][indices]*1.5763e-5)
                    if (true_energies < energy_cut and ion_scint_energies < energy_cut):
                        self.cluster_spectrum[0].append(true_energies)
                        self.cluster_spectrum[1].append(ion_scint_energies)
        fig, axs = plt.subplots()
        if len(self.cluster_spectrum) == 1:
            axs.hist(self.cluster_spectrum[0], bins=num_bins, label=edep_type)
        else:
            axs.hist(self.cluster_spectrum[0], bins=num_bins, label='true', histtype='step', density=True, stacked=True)
            axs.hist(self.cluster_spectrum[1], bins=num_bins, label='ion_scint', histtype='step', density=True, stacked=True)
        axs.set_xlabel('Cluster Energy (MeV)')
        axs.set_xticks(axs.get_xticks() + [6.098])
        axs.set_xlim(0,energy_cut)
        axs.set_title(title)
        plt.legend()
        plt.tight_layout()
        if save != '':
            plt.savefig('plots/'+save+'.png')
        if show:
            plt.show()

    def plot_true_spectrum(self,
        num_bins:   int=100,
        energy_cut: float=10.,   # MeV
        title:  str='Example MC Spectrum',
        save:   str='',
        show:   bool=True,
    ):
        cluster_spectrum = []
        # loop through all edeps
        for ii, truth in enumerate(self.edep_neutron_ids):
            complete = 0
            total = 0
            clusters = np.unique(truth)
            for cluster in clusters:
                indices = np.where(truth==cluster)
                true_energies = sum(self.edep_energy[ii][indices])
                if (true_energies < energy_cut):
                    self.cluster_spectrum.append(true_energies)
        fig, axs = plt.subplots()
        axs.hist(self.cluster_spectrum, bins=num_bins, label='mc spectrum')
        axs.set_xlabel(rf'Capture Energy (MeV) - Complete Capture Ratio ({sum(self.num_complete_captures)}/{sum(self.num_captures)})$\approx${self.capture_ratio}%')
        axs.set_xticks(axs.get_xticks() + [6.098])
        axs.set_xlim(0,energy_cut)
        axs.set_title(title)
        plt.legend()
        plt.tight_layout()
        if save != '':
            plt.savefig('plots/'+save+'.png')
        if show:
            plt.show()

    def scan_truth_eps_values(self,
        eps_range:  list=[1.,100.],
        eps_step:   float=1.,
        save_scores:str='scan_scores'
    ):
        num_steps = int((eps_range[1] - eps_range[0])/eps_step)
        eps_values = [eps_range[0] + ii*eps_step for ii in range(num_steps)]
        scores = [['eps','homogeneity','completeness','v-measure','adjusted_rand_index','adjusted_mutual_info','silhouette']]
        self.logger.info(f"Attempting to run scanning search with {num_steps} eps values from {eps_values[0]} to {eps_values[-1]}.")
        for eps in eps_values:
            self.logger.info(f"Running clustering for eps = {eps}.")
            self.cluster_truth(
                alg = 'dbscan',
                params = {'eps': eps, 'min_samples': 6}
            )
            self.logger.info(f"Computing scores for eps = {eps}.")
            avg_scores = self.calc_truth_scores()
            score_list = [eps]
            for item in avg_scores.keys():
                score_list.append(avg_scores[item])
            scores.append(score_list)
        with open("clustering/" + save_scores + ".csv","w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(scores)
    
    def plot_truth_eps_scores(self,
        input_file,
        save:   str='',
        show:   bool=True
    ):
        scores = pd.read_csv("clustering/" + input_file, delimiter=",")
        if 'eps' not in scores.keys():
            self.logger.error(f"Column for 'eps' values not present in {input_file}: {scores.keys()}")
            raise ValueError(f"Column for 'eps' values not present in {input_file}: {scores.keys()}")
        eps_values = scores['eps']
        fig, axs = plt.subplots(figsize=(8,6))
        for item in scores.keys():
            if item != 'eps':
                axs.plot(
                    eps_values, 
                    scores[item],  
                    linestyle='--'
                )
                max_index = np.argmax(scores[item])
                axs.scatter(
                    eps_values[max_index],
                    scores[item][max_index],
                    label=f'{item} - max ({eps_values[max_index]},{round(scores[item][max_index],3)})',
                    marker='x'
                )
        axs.set_xlabel('eps (mm)')
        axs.set_ylabel('metric')
        axs.set_title('Clustering metric vs. eps (mm)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save != '':
            plt.savefig('plots/'+save+".png")
        if show:
            plt.show()
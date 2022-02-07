"""
Generic analysis code for the unet results
"""
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from unet_logger import UNetLogger
from sklearn import cluster
from sklearn import metrics
import csv

cluster_params = {
    'affinity':     {'damping': 0.5, 'max_iter': 200},
    'mean_shift':   {'bandwidth': None},
    'dbscan':       {'eps': 100.,'min_samples': 6},
    'optics':       {'min_samples': 6},
    'gaussian':     {'n_components': 1, 'covariance_type': 'full', 'tol': 1e-3, 'reg_covar': 1e-6, 'max_iter': 100}
}
class UNetAnalyzer:

    def __init__(self,
        input_file: str,
    ):
        self.logger = UNetLogger('analysis', file_mode='w')
        input   = np.load(input_file, allow_pickle=True)
        self.events  = input['events']
        self.coords  = input['coords']
        self.feats   = input['feats']
        self.energy  = input['energy']
        self.labels  = input['labels']
        self.preds   = input['predictions']
        self.metrics = input['metrics']
        self.metric_names = input['metric_names']

        if len(np.unique(self.labels)) == 2:
            # binary classification
            self.preds = (self.preds > 0.0).astype(int)
            self.correct = (self.preds == self.labels)
            self.label_names = ['neutron','cosmic']
        
    def plot_predictions(self,
        event:  int,
    ):
        if event > len(self.events):
            self.logger.error(f"Event ID {event} greater than number of events: {len(self.events)}.")
            return
        begin = self.events[event][0]
        end   = self.events[event][1]
        coords = self.coords[begin:end]
        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]
        feats  = self.feats[begin:end]
        labels = self.labels[begin:end]
        preds  = self.labels[begin:end]

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
        plt.show()

    def cluster(self,
        level:  str='truth',
        remove_cosmic:  bool=True,
        alg:    str='dbscan',
        params: dict={'eps': 30./4.7,'min_samples': 6},
    ):
        """
        Function for running clustering algorithms on events.
        The level can be ['truth','prediction']
        """
        if level not in ['truth', 'prediction']:
            self.logger.warning(f"Requested cluster level by '{level}' not allowed, using 'truth'.")
            level = 'truth'
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
        self.cluster_predictions = []

        for event in range(len(self.events)):
            begin = self.events[event][0]
            end   = self.events[event][1]
            coords = self.coords[begin:end]
            feats  = self.feats[begin:end]
            labels = self.labels[begin:end]
            preds  = self.labels[begin:end]
            if remove_cosmic:
                coords = coords[(preds == 0)]
            clusterer.fit(coords)
            self.cluster_predictions.append(clusterer.labels_)
        print(self.cluster_predictions)
        
        




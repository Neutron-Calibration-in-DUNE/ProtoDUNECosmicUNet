"""
Generic analysis code for the unet results
"""
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from unet_logger import UNetLogger
import csv


class UNetAnalyzer:

    def __init__(self,
        input_file: str,
    ):
        self.logger = UNetLogger('analysis', file_mode='w')
        input   = np.load(input_file)
        self.events  = input['events']
        self.coords  = input['coords']
        self.feats   = input['feats']
        self.labels  = input['labels']
        self.preds   = input['predictions']
        self.metrics = input['metrics']
        self.metric_names = input['metric_names']
        print(self.metrics)

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

        
        




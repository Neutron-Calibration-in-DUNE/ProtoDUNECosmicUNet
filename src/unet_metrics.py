"""
Metric class for tpc_ml
"""
from collections import OrderedDict
from re import S
import torch
import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, AUC, AUROC
from torchmetrics import MetricCollection
from torchmetrics import Metric
import torch.nn as nn
import numpy as np
import sys
import MinkowskiEngine as ME

from unet_logger import UNetLogger

metric_dict = {
    'accuracy':     Accuracy(),
    'precision':    Precision(),
    'recall':       Recall(),
    'auc':          AUC(),
    'auroc':        AUROC(),
}

# TODO: make a more specialized set of metric classes for different tasks
class Metrics:
    """
    Base class for storing torchmetric metric objects.
    """
    def __init__(self,
        device: str='cpu'
    ):
        self.logger = UNetLogger('metrics', file_mode='w')
        self.logger.info("Running base constructor for metrics type")
        self.metrics = MetricCollection([])
        self.device = device

        self.train_totals = [0. for i in range(len(self.metrics))]
        self.val_totals = [0. for i in range(len(self.metrics))]
        self.test_totals = [0. for i in range(len(self.metrics))]

    def reset_metrics(self):
        self.train_totals = [0. for i in range(len(self.metrics))]
        self.val_totals = [0. for i in range(len(self.metrics))]
        self.test_totals = [0. for i in range(len(self.metrics))]

    def names(self):
        return [item for item in self.metrics.keys()]

    def training_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):
        pass

    def validation_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):
        pass
    
class BinarySemanticSegmentationAccuracy(Metric):
    def __init__(self, 
        dist_sync_on_step=False, 
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("number_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("number_pixels", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self,
        model_output,
        labels
    ):
        model_output = model_output.to(self.device)
        labels = labels.to(self.device)
        predictions = torch.sigmoid(model_output)
        predictions = (predictions > 0.5).float()
        self.number_correct += (predictions == labels).sum()
        self.number_pixels += torch.numel(predictions)

    def compute(self):
        return 100.*self.number_correct/self.number_pixels

class BinarySemanticSegmentationDiceScore(Metric):
    def __init__(self, 
        dist_sync_on_step=False, 
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # for the default state, one must put a floating point number,
        # otherwise it will think the state "dice_score" is of type long
        self.add_state("dice_score", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self,   
        model_output,
        labels
    ):
        model_output = model_output.to(self.device)
        labels = labels.to(self.device)
        predictions = torch.sigmoid(model_output)
        predictions = (predictions > 0.5).float()
        numerator = (2 * (predictions * labels).sum()).float()
        denominator = ((predictions + labels).sum() + 1e-8).float()
        self.dice_score += numerator/denominator

    def compute(self):
        return self.dice_score

class SparseBinarySemanticSegmentationAccuracy(Metric):
    def __init__(self, 
        dist_sync_on_step=False, 
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("number_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("number_pixels", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self,
        model_output,
        labels
    ):
        model_output = model_output.to('cpu')
        labels = labels.to('cpu')
        predictions = torch.sigmoid(model_output)
        predictions = (predictions > 0.5).float()
        self.number_correct += (predictions == labels).sum()
        self.number_pixels += torch.numel(predictions)

    def compute(self):
        return 100.*self.number_correct/self.number_pixels

class SparseBinarySemanticSegmentationDiceScore(Metric):
    def __init__(self, 
        dist_sync_on_step=False, 
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # for the default state, one must put a floating point number,
        # otherwise it will think the state "dice_score" is of type long
        self.add_state("dice_score", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self,   
        model_output,
        labels
    ):
        model_output = model_output.to('cpu')
        labels = labels.to('cpu')
        predictions = torch.sigmoid(model_output)
        predictions = (predictions > 0.5).float()
        numerator = (2 * (predictions * labels).sum()).float()
        denominator = ((predictions + labels).sum() + 1e-8).float()
        self.dice_score += numerator/denominator

    def compute(self):
        return self.dice_score

class BinaryClassificationMetrics(Metrics):
    """
    Class for binary classification metrics
    """
    def __init__(self):
        # construct dictionary of metrics
        super(BinaryClassificationMetrics, self).__init__()
        _dict = OrderedDict()
        _dict['binary_accuracy'] = Accuracy(num_classes=1, average='samples')
        _dict['auc'] = AUROC(num_classes=1, average='weighted')
        self.logger.info(f"Added metric 'binary_accuracy': {Accuracy()} to metrics list.")
        self.logger.info(f"Added metric 'auroc': {AUROC()} to metrics list.")

        # create the collection from the module list
        self.metrics = MetricCollection([
            _dict[item].to(self.device) for item in _dict.keys()
        ])
        self.train_metrics = self.metrics.clone(prefix='train_')
        self.val_metrics = self.metrics.clone(prefix='val_')
        self.test_metrics = self.metrics.clone(prefix='test_')
        self.logger.info(f"Constructed metric collection with items {_dict}.")
        # running totals for a given epoch
        self.reset_metrics()

    def training_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):  
        predictions = predictions.round().to(self.device)
        targets = targets.type('torch.IntTensor').to(self.device)
        results = self.train_metrics(predictions, targets)
        for ii, item in enumerate(results.keys()):
            self.train_totals[ii] += results[item]/float(batches)
        
    def validation_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):  
        predictions = predictions.round().to(self.device)
        targets = targets.type('torch.IntTensor').to(self.device)
        results = self.val_metrics(predictions, targets)
        for ii, item in enumerate(results.keys()):
            self.val_totals[ii] += results[item]/float(batches)
    
    def testing_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):  
        predictions = predictions.round().to(self.device)
        targets = targets.type('torch.IntTensor').to(self.device)
        results = self.test_metrics(predictions, targets)
        for ii, item in enumerate(results.keys()):
            self.test_totals[ii] += results[item]/float(batches)

class MulticlassClassificationMetrics(Metrics):
    """
    Class for Multiclass classification metrics
    """
    def __init__(self,
        num_classes,
    ):
        # construct dictionary of metrics
        super(MulticlassClassificationMetrics, self).__init__()
        self.num_classes = num_classes
        self.logger.info(f"creating multiclass classification metrics with {self.num_classes} classes")
        _dict = OrderedDict()
        _dict['accuracy'] = Accuracy(num_classes=self.num_classes, average='samples')
        _dict['auc'] = AUROC(num_classes=self.num_classes, average='weighted')
        self.logger.info(f"Added metric 'accuracy': {Accuracy()} to metrics list.")
        self.logger.info(f"Added metric 'auroc': {AUROC()} to metrics list.")

        # create the collection from the module list
        self.metrics = MetricCollection([
            _dict[item].to(self.device) for item in _dict.keys()
        ])
        self.train_metrics = self.metrics.clone(prefix='train_')
        self.val_metrics = self.metrics.clone(prefix='val_')
        self.test_metrics = self.metrics.clone(prefix='test_')
        self.logger.info(f"Constructed metric collection with items {_dict}.")
        # running totals for a given epoch
        self.reset_metrics()

    def training_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):  
        predictions = predictions.to(self.device)
        targets = targets.type('torch.IntTensor').to(self.device)
        results = self.train_metrics(predictions, targets)
        for ii, item in enumerate(results.keys()):
            self.train_totals[ii] += results[item]/float(batches)
        
    def validation_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):  
        predictions = predictions.to(self.device)
        targets = targets.type('torch.IntTensor').to(self.device)
        results = self.val_metrics(predictions, targets)
        for ii, item in enumerate(results.keys()):
            self.val_totals[ii] += results[item]/float(batches)
    
    def testing_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):  
        predictions = predictions.to(self.device)
        targets = targets.type('torch.IntTensor').to(self.device)
        results = self.test_metrics(predictions, targets)
        for ii, item in enumerate(results.keys()):
            self.test_totals[ii] += results[item]/float(batches)

class BinarySemanticSegmentationMetrics(Metrics):
    """
    Class for binary SemanticSegmentation metrics
    """
    def __init__(self):
        # construct dictionary of metrics
        super(BinarySemanticSegmentationMetrics, self).__init__()
        _dict = OrderedDict()
        _dict['binary_accuracy'] = BinarySemanticSegmentationAccuracy()
        _dict['dice_score'] = BinarySemanticSegmentationDiceScore()
        self.logger.info(f"Added metric 'binary_accuracy': {BinarySemanticSegmentationAccuracy()} to metrics list.")
        self.logger.info(f"Added metric 'dice_score': {BinarySemanticSegmentationDiceScore()} to metrics list.")

        # create the collection from the module list
        self.metrics = MetricCollection([
            _dict[item].to(self.device) for item in _dict.keys()
        ])
        self.train_metrics = self.metrics.clone(prefix='train_')
        self.val_metrics = self.metrics.clone(prefix='val_')
        self.test_metrics = self.metrics.clone(prefix='test_')
        self.logger.info(f"Constructed metric collection with items {_dict}.")
        # running totals for a given epoch
        self.reset_metrics()

    def training_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):  
        predictions = predictions.to(self.device)
        targets = targets.type('torch.IntTensor').to(self.device)
        results = self.train_metrics(predictions, targets)
        for ii, item in enumerate(results.keys()):
            self.train_totals[ii] += results[item]/float(batches)
        
    def validation_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):  
        predictions = predictions.to(self.device)
        targets = targets.type('torch.IntTensor').to(self.device)
        results = self.val_metrics(predictions, targets)
        for ii, item in enumerate(results.keys()):
            self.val_totals[ii] += results[item]/float(batches)
    
    def testing_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):  
        predictions = predictions.to(self.device)
        targets = targets.type('torch.IntTensor').to(self.device)
        results = self.test_metrics(predictions, targets)
        for ii, item in enumerate(results.keys()):
            self.test_totals[ii] += results[item]/float(batches)

class SparseBinarySemanticSegmentationMetrics(Metrics):
    """
    Class for binary SemanticSegmentation metrics
    """
    def __init__(self):
        # construct dictionary of metrics
        super(SparseBinarySemanticSegmentationMetrics, self).__init__()
        _dict = OrderedDict()
        _dict['binary_accuracy'] = SparseBinarySemanticSegmentationAccuracy()
        _dict['dice_score'] = SparseBinarySemanticSegmentationDiceScore()
        self.logger.info(f"Added metric 'binary_accuracy': {SparseBinarySemanticSegmentationAccuracy()} to metrics list.")
        self.logger.info(f"Added metric 'dice_score': {SparseBinarySemanticSegmentationDiceScore()} to metrics list.")

        # create the collection from the module list
        self.metrics = MetricCollection([
            _dict[item].to(self.device) for item in _dict.keys()
        ])
        self.train_metrics = self.metrics.clone(prefix='train_')
        self.val_metrics = self.metrics.clone(prefix='val_')
        self.test_metrics = self.metrics.clone(prefix='test_')
        self.logger.info(f"Constructed metric collection with items {_dict}.")
        # running totals for a given epoch
        self.reset_metrics()

    def training_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):  
        predictions = predictions.F.to(self.device)
        targets = targets.type('torch.IntTensor').to(self.device)
        results = self.train_metrics(predictions, targets)
        for ii, item in enumerate(results.keys()):
            self.train_totals[ii] += results[item]/float(batches)
        
    def validation_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):  
        predictions = predictions.F.to(self.device)
        targets = targets.type('torch.IntTensor').to(self.device)
        results = self.val_metrics(predictions, targets)
        for ii, item in enumerate(results.keys()):
            self.val_totals[ii] += results[item]/float(batches)
    
    def testing_metrics(self,
        predictions,
        targets,
        batches:    int=1.0,
    ):  
        predictions = predictions.F.to(self.device)
        targets = targets.type('torch.IntTensor').to(self.device)
        results = self.test_metrics(predictions, targets)
        for ii, item in enumerate(results.keys()):
            self.test_totals[ii] += results[item]/float(batches)
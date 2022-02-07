"""
Generic trainer 
"""
import os
import torch
import torch.optim as optim
import torch.nn as nn
from unet_metrics import Metrics
from tqdm import tqdm
import time
import sys
import numpy as np
import MinkowskiEngine as ME
import warnings
warnings.filterwarnings("ignore")

from unet_logger import UNetLogger

# TODO: create inference function
class SparseTrainer:
    """
    Generic trainer class for generic model types.
    
    You must pass in 
        1.  model - a nn.Module instance describing the model
        2.  criterion - the loss function you want to use
        3.  optimizer - the optimization algorithm you want to use
    
    Other optional inputs include:
        4.  metrics - a metric object that contains the desired metrics to use
        5.  gpu - whether to use a gpu
        6.  gpu_device - the number associated to the gpu you want to use
        7.  seed - a random seed to use for reproducibility in pytorch
    """
    def __init__(self,
        model,
        criterion,
        optimizer,
        metrics:        Metrics=None,
        metric_type:    str='test',
        gpu:            bool=True,
        gpu_device:     int=0,
        seed:           int=0,
        multi_class:    bool=False,
        semantic_segmentation:  bool=False,
    ): 
        self.logger = UNetLogger('trainer', file_mode='w')
        self.training_logger = UNetLogger('training_epoch', output='training_epoch')
        self.validation_logger = UNetLogger('validation_epoch', output='validation_epoch')
        self.training_metrics_logger = UNetLogger('training_metrics', output='training_metrics')
        self.validation_metrics_logger = UNetLogger('validation_metrics', output='validation_metrics')
        self.testing_logger = UNetLogger('testing', output='testing')
        self.testing_metrics_logger = UNetLogger('testing_metrics', output='testing_metrics')
        if metric_type not in ['train','test','both']:
            self.logger.warn(f"Metric type {metric_type} not in {['train','test','both']}.  Setting type to 'test'.")
            metric_type = 'test'
        self.metric_type = metric_type
        if not os.path.isdir('predictions/'):
            os.mkdir('predictions/')
        # first initialize the trainer
        self.logger.info("Initializing SparseTrainer...")
        self.model = model
        self.gpu = gpu
        self.gpu_device = gpu_device
        self.seed = seed
        self.multi_class = multi_class
        self.semantic_segmentation = semantic_segmentation
        if self.multi_class == True:
            self.logger.info(f"Setting multi_class to true means that classes should be specified as ints")
        if self.semantic_segmentation == True:
            self.logger.info(f"Setting semantic segmentation argument to true.")
        # check for devices
        if torch.cuda.is_available():
            self.logger.info(f"CUDA is available with devices:")
            for ii in range(torch.cuda.device_count()):
                device_properties = torch.cuda.get_device_properties(ii)
                cuda_stats = f"name: {device_properties.name}, "
                cuda_stats += f"compute: {device_properties.major}.{device_properties.minor}, "
                cuda_stats += f"memory: {device_properties.total_memory}"
                self.logger.info(f" -- device: {ii} - " + cuda_stats)

        # set gpu settings
        if self.gpu:
            if torch.cuda.is_available():
                if gpu_device >= torch.cuda.device_count() or gpu_device < 0:
                    self.logger.warn(f"Desired gpu_device '{gpu_device}' not available, using device '0'")
                    self.gpu_device = 0
                self.device = torch.device(f"cuda:{self.gpu_device}")
                self.logger.info(f"CUDA is available, using device {self.gpu_device} - {torch.cuda.get_device_name(self.gpu_device)}")
            else:
                self.gpu == False
                self.logger.warn(f"CUDA not available! Using the cpu")
                self.device = torch.device("cpu")
        else:
            self.logger.info(f"Using cpu as device")
            self.device = torch.device("cpu")
        self.model.to(self.device)

        # set the loss function to use
        self.criterion = criterion.criterion
        self.criterion_name = criterion.name
        # set the optimizer
        self.optimizer = optimizer
        # collect the metrics
        self.metrics = metrics
        if self.metrics == None:
            self.metric_names = []
        else:
            self.metric_names = self.metrics.names()
        # move metrics to device
        self.metrics.metrics.to(self.device)
        self.metrics.device = self.device

        # add timing info
        if self.gpu:
            self.train_timer_start = torch.cuda.Event(enable_timing=True)
            self.train_timer_end   = torch.cuda.Event(enable_timing=True)
            self.val_timer_start = torch.cuda.Event(enable_timing=True)
            self.val_timer_end   = torch.cuda.Event(enable_timing=True)
        else:
            self.train_timer_start = 0
            self.train_timer_end = 0
            self.val_timer_start = 0
            self.val_timer_end = 0

    # TODO: allow changes of training parameters on input
    # TODO: this means allowing the validation set to be changed?
    # TODO: save final state_dict as array, have the option to save to numpy file

    def train(self,
        dataset_loader,             # dataset_loader to pass in
        dataset_type   = 'train',   # train, test or all
        epochs      = 100,          # number of epochs to train
        validation  = .3,           # amount to hold aside for validation
        shuffle_validation = False, # whether to shuffle the validation each epoch
        checkpoint  = 10,           # epochs inbetween weight saving
        log_step    = 10,           # epochs inbetween each metric evaluation
        progress_bar:   str='train',# progress bar from tqdm
        rewrite_bar:    bool=False, # wether to leave the bars after each epoch
    ):
        """
        Training function.
        """
        # TODO: check that model is consistent with dataset_loader
        
        # iterate over epochs
        self.logger.info(f"Training model for {epochs} epochs...")
        # Training
        for epoch in range(epochs):
            # set up losses and metrics for epoch
            training_loss   = 0.0
            validation_loss = 0.0
            
            if self.gpu:
                self.train_timer_start.record()
            else:
                self.train_timer_start = time.time()

            # run the training loop
            if (progress_bar == 'all' or progress_bar == 'train'):
                training_loop = tqdm(
                    enumerate(dataset_loader.train_loader, 0), 
                    total=len(dataset_loader.train_loader), 
                    leave=rewrite_bar,
                    colour='green'
                )
            else:
                training_loop = enumerate(dataset_loader.train_loader, 0)

            # make sure to set model to train() during training!
            self.model.train()
            for ii, data in training_loop:
                ## Set up input #######################################
                # get the inputs
                if dataset_loader.weights == False:
                    coords, feats, labels = data
                else:
                    coords, feats, labels, weights = data
                    weights.to(self.device)
                # check if ss problem
                if not self.semantic_segmentation:
                    # check if this is a binary classification problem
                    if not self.multi_class:
                        labels = labels.squeeze(1).float()
                    else:
                        labels = labels.squeeze(1).to(torch.int64)
                else:
                    if not self.multi_class:
                        labels = labels.float()
                ## Forward + Backward + Optimize ###################### 
                # zero the parameter gradients
                self.optimizer.zero_grad()
                outputs = self.model(ME.SparseTensor(feats.float(), coords, device=self.device))
                loss = self.criterion(outputs.F.squeeze(), labels.to(self.device))
                if dataset_loader.weights == True:
                    loss = (loss * weights.to(self.device) / weights.sum().to(self.device)).sum()
                loss.backward()
                self.optimizer.step()

                # accumulate losses
                training_loss += loss.item() / dataset_loader.num_train_batches
                # compute metrics
                if self.metrics != None and (self.metric_type == 'train' or self.metric_type == 'both'):
                    self.metrics.training_metrics(outputs, labels, dataset_loader.num_train_batches)
                if (progress_bar == 'all' or progress_bar == 'train'):
                    training_loop.set_description(f"Training: Epoch [{epoch}/{epochs}]")
                    training_loop.set_postfix(loss = loss.item())

            if self.gpu:
                self.train_timer_end.record()
                self.val_timer_start.record()
            else:
                self.train_time_end = time.time()
                self.val_timer_start = time.time()

            # run the validation loop
            if (progress_bar == 'all' or progress_bar == 'val'):
                validation_loop = tqdm(
                    enumerate(dataset_loader.validation_loader, 0),
                    total=len(dataset_loader.validation_loader),
                    leave=False
                )
            else:
                validation_loop = enumerate(dataset_loader.validation_loader, 0)

            # make sure to set model to eval() during validation!
            self.model.eval()
            for ii, data in validation_loop:
                ## Set up input #######################################
                # get the inputs
                if dataset_loader.weights == False:
                    coords, feats, labels = data
                else:
                    coords, feats, labels, weights = data
                    weights.to(self.device)
                # check if ss problem
                if not self.semantic_segmentation:
                    # check if this is a binary classification problem
                    if not self.multi_class:
                        labels = labels.squeeze(1).float()
                    else:
                        labels = labels.squeeze(1).to(torch.int64)
                else:
                    if not self.multi_class:
                        labels = labels.float()
                ## Forward + Backward + Optimize ###################### 
                with torch.no_grad():
                    outputs = self.model(ME.SparseTensor(feats.float(), coords, device=self.device))
                    loss = self.criterion(outputs.F.squeeze(), labels.to(self.device))
                    if dataset_loader.weights == True:
                        loss = (loss * weights.to(self.device) / weights.sum().to(self.device)).sum()
                    validation_loss += loss.item() / dataset_loader.num_val_batches
                # compute metrics
                if self.metrics != None and (self.metric_type == 'train' or self.metric_type == 'both'):
                    self.metrics.validation_metrics(outputs, labels, dataset_loader.num_val_batches)
                if (progress_bar == 'all' or progress_bar == 'val'):
                    validation_loop.set_description(f"Validation: Epoch [{epoch}/{epochs}]")
                    validation_loop.set_postfix(loss = loss.item())

            # collect statistics from epoch
            if self.gpu:
                self.val_timer_end.record()
                torch.cuda.synchronize()
                self.training_logger.epoch(
                    epoch,
                    self.train_timer_start.elapsed_time(self.train_timer_end),
                    training_loss
                )
                self.validation_logger.epoch(
                    epoch,
                    self.val_timer_start.elapsed_time(self.val_timer_end),
                    validation_loss
                )
            else:
                self.val_timer_end = time.time()
                self.training_logger.epoch(
                    epoch,
                    self.train_timer_end - self.train_timer_start,
                    training_loss
                )
                self.validation_logger.epoch(
                    epoch,
                    self.val_timer_end - self.val_timer_start,
                    validation_loss
                )

            if self.metrics != None and (self.metric_type == 'train' or self.metric_type == 'both'):
                self.training_metrics_logger.metrics(epoch,self.metrics.train_totals)
                self.validation_metrics_logger.metrics(epoch,self.metrics.val_totals)

            # save weights if at checkpoint step
            if epoch % checkpoint == 0:
                if not os.path.exists(".checkpoints/"):
                    os.makedirs(".checkpoints/")
                torch.save(self.model.state_dict(), f".checkpoints/checkpoint_{epoch}.ckpt")
            # save metrics if at log step
            if epoch % log_step:
                pass

        self.logger.info("Finished training.")
        # # TODO: save final model as checkpoint and in datatypes array
        # # TODO: save metrics as array in datatypes npz

        # run through test samples
        test_loss = 0.0
        if (progress_bar == 'all' or progress_bar == 'test'):
            test_loop = tqdm(
                enumerate(dataset_loader.test_loader, 0),
                total=len(dataset_loader.test_loader),
                leave=False
            )
        else:
            test_loop = enumerate(dataset_loader.test_loader, 0)
        self.model.eval()
        for ii, data in test_loop:
            ## Set up input #######################################
            # get the inputs
            if dataset_loader.weights == False:
                coords, feats, labels = data
            else:
                coords, feats, labels, weights = data
                weights.to(self.device)
            # check if ss problem
            if not self.semantic_segmentation:
                # check if this is a binary classification problem
                if not self.multi_class:
                    labels = labels.squeeze(1).float()
                else:
                    labels = labels.squeeze(1).to(torch.int64)
            else:
                if not self.multi_class:
                    labels = labels.float()

            ## Forward + Backward + Optimize ###################### 
            with torch.no_grad():
                outputs = self.model(ME.SparseTensor(feats.float(), coords, device=self.device))
                loss = self.criterion(outputs.F.squeeze(), labels.to(self.device))
                if dataset_loader.weights == True:
                    loss = (loss * weights.to(self.device) / weights.sum().to(self.device)).sum()
                test_loss += loss.item() / dataset_loader.num_test_batches

            # compute metrics
            if self.metrics != None and (self.metric_type == 'test' or self.metric_type == 'both'):
                self.metrics.testing_metrics(outputs, labels, dataset_loader.num_test_batches)
            if (progress_bar == 'all' or progress_bar == 'test'):
                test_loop.set_description(f"test: Epoch [{epoch}/{epochs}]")
                test_loop.set_postfix(loss = loss.item())
        self.testing_logger.epoch(
            epoch,
            0,
            test_loss
        )
        if self.metrics != None and (self.metric_type == 'test' or self.metric_type == 'both'):
            self.testing_metrics_logger.metrics(epoch,self.metrics.test_totals)
    
    def inference(self,
        dataset_loader,                     # dataset_loader to pass in
        dataset_type:       str='train',    # train, val, test or all
        save_predictions:   bool=True,
        output_file:        str='output.npz',
    ):
        """
        Inference loop
        """
        # if dataset_type == 'train':
        #     inference_loop = enumerate(dataset_loader.train_loader, 0)
        # elif dataset_type == 'val':
        #     inference_loop = enumerate(dataset_loader.validation_loader, 0)
        # elif dataset_type == 'test':
        #     inference_loop = enumerate(dataset_loader.test_loader, 0)
        # elif dataset_type == 'all':
        #     inference_loop = enumerate(dataset_loader.all_loader, 0)
        # else:
        #     self.logger.warn(f"dataset_type '{dataset_type}' is an invalid choice.  valid choices are 'train', 'val', 'test' and 'all', setting dataset_type to 'all'.")
        #     dataset_type = 'all'
        #     inference_loop = enumerate(dataset_loader.all_loader, 0)
        inference_loop = enumerate(dataset_loader.inference_loader, 0)
        self.logger.info(f"running inference on '{dataset_type}' dataset")

        inference_loss = 0.0
        self.model.eval()
        # output variables
        saved_coords = np.empty((1,4), dtype=np.int32)
        saved_feats = np.empty((1,1))
        saved_labels = np.empty(1)
        saved_predictions = np.empty((1,1))
        saved_events = []
        saved_metrics = []
        # run through the inference loop
        for ii, data in inference_loop:
            # get the inputs
            if dataset_loader.weights == False:
                coords, feats, labels = data
            else:
                coords, feats, labels, weights = data
                weights.to(self.device)
            # check if ss problem
            if not self.semantic_segmentation:
                # check if this is a binary classification problem
                if not self.multi_class:
                    labels = labels.squeeze(1).float()
                else:
                    labels = labels.squeeze(1).to(torch.int64)
            else:
                if not self.multi_class:
                    labels = labels.float()
            # forward + backward
            with torch.no_grad():
                outputs = self.model(ME.SparseTensor(feats.float(), coords, device=self.device))
                # loss = self.criterion(outputs.F.squeeze(), labels.to(self.device))
                # if dataset_loader.weights == True:
                #     loss = (loss * weights.to(self.device) / weights.sum().to(self.device)).sum()
                # if dataset_type == 'train':
                #     inference_loss += loss.item() / dataset_loader.num_train_batches
                # elif dataset_type == 'val':
                #     inference_loss += loss.item() / dataset_loader.num_val_batches
                # elif dataset_type == 'test':
                #     inference_loss += loss.item() / dataset_loader.num_test_batches
                # else:
                #     inference_loss += loss.item() / dataset_loader.num_all_batches
                
                saved_coords = np.concatenate((saved_coords, coords.cpu()))
                saved_feats  = np.concatenate((saved_feats, feats.cpu()))
                saved_labels = np.concatenate((saved_labels, labels.cpu()))
                saved_predictions = np.concatenate((saved_predictions, outputs.F.cpu()))
                saved_metrics.append(self.metrics.inference_metrics(outputs, labels, 1.0))
                if ii == 0:
                    saved_events.append([0,len(coords)-1])
                else:
                    saved_events.append([saved_events[ii-1][1]+1,saved_events[ii-1][1] + len(coords)])
                if ii == 0:
                    saved_coords = np.delete(saved_coords, 0, 0)
                    saved_feats = np.delete(saved_feats, 0, 0)
                    saved_labels = np.delete(saved_labels, 0, 0)
                    saved_predictions = np.delete(saved_predictions, 0, 0)
            
        saved_coords = np.delete(saved_coords, 0, 1)
        saved_feats = saved_feats.flatten()
        saved_labels = saved_labels.flatten()
        saved_predictions = saved_predictions.flatten()
        if save_predictions:
            np.savez(
                'predictions/'+output_file,
                events=saved_events,
                coords=saved_coords,
                feats=saved_feats,
                labels=saved_labels,
                predictions=saved_predictions,
                energy=dataset_loader.dataset.energy,
                metrics=saved_metrics,
                metric_names=self.metrics.metric_names,
            )
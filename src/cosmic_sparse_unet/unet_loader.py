"""
SparseUNet data loader
"""
from torch.utils.data import DataLoader, random_split
import MinkowskiEngine as ME
import sys

from unet_logger import UNetLogger

class SparseUNetLoader():

    def __init__(self,
        dataset,
        batch_size,
        validation_split:   float=0.0,
        test_split:         float=0.2,
        num_workers:        int=0,
    ):
        self.logger = UNetLogger('loader', file_mode='w')
        self.logger.info("attempting to construct data loader")
        self.logger.info(f"data loader batch size = {batch_size}")
        self.logger.info(f"data loader validation split = {validation_split}")
        self.logger.info(f"data loader num_workers = {num_workers}")
        self.logger.info(f"data loader test split = {test_split}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.num_workers = num_workers
        if self.dataset.weights != None:
            self.weights = True
        else:
            self.weights = False
        
        # determine number of training/testing samples
        self.total_train = int(len(self.dataset) * (1 - self.test_split))
        self.num_test  = int(len(self.dataset) - self.total_train)
        self.num_test_batches = int(self.num_test/self.batch_size)
        if self.num_test % self.batch_size != 0:
            self.num_test_batches += 1
        self.logger.info(f"data loader number of total training samples: {self.total_train}")
        self.logger.info(f"data loader number of test samples: {self.num_test}")
        self.logger.info(f"data loader number of test batches: {self.num_test_batches}")

        # determine number of training/validation samples
        self.num_train  = int(self.total_train * (1 - self.validation_split))
        self.num_val    = int(self.total_train - self.num_train)
        self.num_train_batches = int(self.num_train/self.batch_size)
        if self.num_train % self.batch_size != 0:
            self.num_train_batches += 1
        self.num_val_batches   = int(self.num_val/self.batch_size)
        if self.num_val % self.batch_size != 0:
            self.num_val_batches += 1

        self.logger.info(f"data loader number of training samples: {self.num_train}")
        self.logger.info(f"data loader number of training batches per epoch: {self.num_train_batches}")
        self.logger.info(f"data loader number of validation samples: {self.num_val}")
        self.logger.info(f"data loader number of validation batches per epoch: {self.num_val_batches}")

        # set up the training and testing sets
        self.total_train, self.test = random_split(self.dataset, [self.total_train, self.num_test])
        # set up the training and validation sets
        self.train, self.validation = random_split(self.total_train, [self.num_train, self.num_val])
        # set up dataloaders for each
        self.train_loader = DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=ME.utils.SparseCollation())

        self.validation_loader = DataLoader(
            self.validation, 
            batch_size=self.batch_size, 
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=ME.utils.SparseCollation())

        self.test_loader = DataLoader(
            self.test, 
            batch_size=self.batch_size, 
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=ME.utils.SparseCollation())

        self.all_loader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=ME.utils.SparseCollation())
        
        self.inference_loader = DataLoader(
            self.dataset,
            batch_size=1,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=ME.utils.SparseCollation())
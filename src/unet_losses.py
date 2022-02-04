"""
"""
from unet_logger import UNetLogger
import torch.nn as nn

class GenericLoss:

    def __init__(self,
        criterion:  str='MSELoss'
    ):
        self.logger = UNetLogger('losses', file_mode='w')
        # set the criterion
        self.name = criterion
        # TODO: need checks here for compatibility of the model and the loss
        if criterion == 'MSELoss':
            self.criterion = nn.MSELoss()
            self.logger.info("Using mean squared error loss")
        elif criterion == 'BCELoss':
            self.criterion = nn.BCELoss()
            self.logger.info("Using binary cross entropy loss")
        elif criterion == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
            self.logger.info("Using cross entropy loss")
        elif criterion == 'BCEWithLogitsLoss':
            self.criterion = nn.BCEWithLogitsLoss()
            self.logger.info("Using binary cross entropy loss with logits")
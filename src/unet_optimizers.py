"""
"""
from unet_logger import UNetLogger
import torch.optim as optim

class GenericOptimizer:

    def __init__(self,
        model,
        optimizer:  str='Adam',
        learning_rate:  float=0.001,
        momentum:   float=0.9
    ):
        self.logger = UNetLogger('optimizers', file_mode='w')
        # set learning rate and momentum
        self.learning_rate = learning_rate
        self.logger.info(f"Learning rate set to {self.learning_rate}")
        self.momentum = momentum
        self.logger.info(f"Momentum value set to {self.momentum}")

        # set the optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
            )
            self.logger.info("Using the Adam optimizer")
        
    def zero_grad(self):
        return self.optimizer.zero_grad()
    
    def step(self):
        return self.optimizer.step()
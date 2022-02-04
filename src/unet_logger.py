"""
Global logger for the UNet
"""
import logging
from multiprocessing.sharedctypes import Value
import sys
import os

logging_level = {
    'debug':    logging.DEBUG,
}

logging_output = [
    'console',
    'file',
    'both',
    'training_epoch',
    'validation_epoch',
    'testing',
    'training_metrics',
    'validation_metrics',
    'testing_metrics',
]


class UNetLogger:
    """
    Logger for UNet related classes
    """

    def __init__(self,
        name:   str='default',
        level:  str='debug',
        output: str='file',
        output_file:    str='',
        file_mode:  str='a',
    ):
        self.name = name
        if level not in logging_level.keys():
            raise ValueError(f"Logging level {level} not in {logging_level}.")
        if output not in logging_output:
            raise ValueError(f"Logging handler {output} not in {logging_output}.")
        if not os.path.isdir('.logs'):
            os.mkdir('.logs')
        self.level = logging_level[level]
        self.output = output
        if output_file == '':
            self.output_file = name
        else:
            self.output_file = output_file
        self.file_mode = file_mode
        # create logger
        self.logger = logging.getLogger(self.name)
        # set level
        self.logger.setLevel(self.level)
        # set format
        self.console_formatter = logging.Formatter('[%(levelname)s]: %(message)s')
        self.file_formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s')
        self.training_formatter = logging.Formatter('%(message)s')
        self.validation_formatter = logging.Formatter('%(message)s')
        self.testing_formatter = logging.Formatter('%(message)s')
        # create handler
        if self.output == 'console' or self.output == 'both':
            self.console = logging.StreamHandler()
            self.console.setLevel(self.level)
            self.console.setFormatter(self.formatter)
            self.logger.addHandler(self.console)
        if self.output == 'file' or self.output == 'both':
            self.file = logging.FileHandler('.logs/'+self.output_file+'.log', mode=self.file_mode)
            self.file.setLevel(self.level)
            self.file.setFormatter(self.file_formatter)
            self.logger.addHandler(self.file)
        if self.output == 'training_epoch':
            self.training_file = logging.FileHandler('.logs/training_epoch.log', mode='w')
            self.training_file.setLevel(self.level)
            self.training_file.setFormatter(self.training_formatter)
            self.logger.addHandler(self.training_file)
            self.logger.info('epoch,time,loss')
        if self.output == 'validation_epoch':
            self.validation_file = logging.FileHandler('.logs/validation_epoch.log', mode='w')
            self.validation_file.setLevel(self.level)
            self.validation_file.setFormatter(self.validation_formatter)
            self.logger.addHandler(self.validation_file)
            self.logger.info('epoch,time,loss')
        if self.output == 'testing':
            self.testing_file = logging.FileHandler('.logs/testing.log', mode='w')
            self.testing_file.setLevel(self.level)
            self.testing_file.setFormatter(self.testing_formatter)
            self.logger.addHandler(self.testing_file)
            self.logger.info('epoch,time,loss')
        if self.output == 'training_metrics':
            self.training_file = logging.FileHandler('.logs/training_metrics.log', mode='w')
            self.training_file.setLevel(self.level)
            self.training_file.setFormatter(self.training_formatter)
            self.logger.addHandler(self.training_file)
        if self.output == 'validation_metrics':
            self.validation_file = logging.FileHandler('.logs/validation_metrics.log', mode='w')
            self.validation_file.setLevel(self.level)
            self.validation_file.setFormatter(self.validation_formatter)
            self.logger.addHandler(self.validation_file)
        if self.output == 'testing_metrics':
            self.testing_file = logging.FileHandler('.logs/testing_metrics.log', mode='w')
            self.testing_file.setLevel(self.level)
            self.testing_file.setFormatter(self.testing_formatter)
            self.logger.addHandler(self.testing_file)

    def info(self,
        message:    str,
    ):
        return self.logger.info(message)
    
    def debug(self,
        message:    str,
    ):
        return self.logger.debug(message)

    def warn(self,
        message:    str,
    ):
        return self.logger.warning(message)

    def error(self,
        message:    str,
    ):
        return self.logger.error(message)
    
    def epoch(self,
        epoch:  int,
        time:   float,
        loss:   float,
    ):
        return self.logger.info(f'{epoch},{time},{loss}')
    
    def metrics(self,
        epoch:  int,
        metrics: list,
    ):
        message = str(epoch)
        for metric in metrics:
            message += ','+str(metric)
        return self.logger.info(message)

import os
import math
import shutil
from abc import abstractmethod

import yaml
import torch

from log.logger import setup_logger


class TrainerBase:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metrics, optimizer, config, device):
        self.logger = setup_logger(self, verbose=config.TRAIN.VERBOSE)
        self.model = model
        self.device = device
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.config = config

        self.epochs = config.TRAIN.MAX_EPOCHS
        self.save_period = config.TRAIN.SAVE_PERIOD

        self.start_epoch = 1

        self.checkpoint_dir = config.TRAIN.SAVE_DIR

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        self.logger.info('Starting training...')
        for epoch in range(self.start_epoch, self.epochs):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            log.update(result)
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'value_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value
            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info(f'{str(key):15s}: {value}')

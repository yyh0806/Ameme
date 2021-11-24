import torch
from abc import abstractmethod
from loguru import logger

from ame.meter import MetricTracker


class TrainerBase:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metrics, optimizer, epoch, device, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, checkpoint=None, save_dir=None):

        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.epochs = epoch
        self.device = device
        self.start_epoch = 1
        self.save_dir = save_dir
        self.save_period = 1
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics])
        if checkpoint is not None:
            self._resume_checkpoint(checkpoint)

    def train(self):
        logger.info('Starting training...')

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            log.update(train_result)
            if self.valid_data_loader:
                valid_result = self._valid_epoch(epoch)
                log.update(valid_result)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(valid_result['loss'])

    @abstractmethod
    def _train_epoch(self, epoch: int) -> dict:

        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch: int) -> dict:

        raise NotImplementedError

    @abstractmethod
    def _calculate_loss(self, data, target):
        raise NotImplementedError(
            "calculate_loss should be implemented by subclass!")

    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        filename = self.save_dir + "/" + f'checkpoint-epoch{epoch}.pth'
        torch.save(state, filename)
        logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = self.save_dir + "/" + 'model_best.pth'
            torch.save(state, best_path)
            logger.info(f'Saving current best: {best_path}')

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

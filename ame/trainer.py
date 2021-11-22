import sys
import torch
from base import TrainerBase
from tqdm import tqdm


class Trainer(TrainerBase):

    def __init__(self, model, criterion, metrics, optimizer, epoch, device, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, checkpoint=None, save_dir=None):
        super(Trainer, self).__init__(model, criterion, metrics, optimizer, epoch, device,
                                      train_data_loader, valid_data_loader, lr_scheduler, checkpoint, save_dir)

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.train_metrics.reset()
        with tqdm(self.train_data_loader, desc="train", file=sys.stdout) as iterator:
            for data, target in iterator:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self._calculate_loss(output, target)
                loss.backward()
                self.optimizer.step()

                self.train_metrics.update('loss', loss.item())
                for met in self.metrics:
                    self.train_metrics.update(met.__name__, met(output, target))

                s = self._format_logs(self.train_metrics.result())
                iterator.set_postfix_str(s)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            return self.train_metrics.result()

    def _valid_epoch(self, epoch: int) -> dict:
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            with tqdm(self.valid_data_loader, desc="valid", file=sys.stdout) as iterator:
                for data, target in iterator:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self._calculate_loss(output, target)

                    self.valid_metrics.update('loss', loss.item())
                    for met in self.metrics:
                        self.valid_metrics.update(met.__name__, met(output, target))
                return self.valid_metrics.result()

    def _calculate_loss(self, data, target):
        return self.criterion(data, target)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s
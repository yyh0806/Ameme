import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from ame.base.base_dataloader import DataLoaderBase
from ame.dataset.datasets import *
from ame.dataset.augmentation import *
import pandas as pd


class CellDataLoader(DataLoaderBase):

    def __init__(self, data_path, batch_size, shuffle=True, validation_split=0.0, num_workers=0):
        transforms = CellTransforms()
        self.train_dataset = CellDataset(data_path, transforms)
        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers
        }
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers)

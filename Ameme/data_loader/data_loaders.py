import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import os
import pandas as pd
from Ameme.base import DataLoaderBase


class MnistDataLoader(DataLoaderBase):
    """
    MNIST data loading demo using DataLoaderBase
    """

    def __init__(self, transforms, data_dir, batch_size, shuffle, validation_split, nworkers,
                 train=True):
        self.data_dir = data_dir

        self.train_dataset = datasets.MNIST(
            self.data_dir,
            train=train,
            download=True,
            transform=transforms.build_transforms(train=True)
        )
        self.valid_dataset = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.build_transforms(train=False)
        ) if train else None

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers
        }
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)

    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset, **self.init_kwargs)


class SevenSegmentDataLoader(DataLoaderBase):

    def __init__(self, transforms, data_dir, batch_size, shuffle, validation_split, nworkers, train=True):
        self.data_dir = data_dir

        self.train_dataset = datasets.ImageFolder(
            self.data_dir,
            transform=transforms.build_transforms(train=True)
        )
        self.valid_dataset = datasets.ImageFolder(
            self.data_dir,
            transform=transforms.build_transforms(train=False)
        ) if train else None

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers
        }
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)

    def __len__(self):
        return len(self.data)

    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset, **self.init_kwargs)


class NissinDataLoader(DataLoaderBase):

    def __init__(self, transforms, data_dir, batch_size, shuffle, validation_split, nworkers, train=True):
        self.data_dir = data_dir

        self.train_dataset = datasets.ImageFolder(
            self.data_dir,
            transform=transforms.build_transforms(train=True)
        )
        self.valid_dataset = datasets.ImageFolder(
            self.data_dir,
            transform=transforms.build_transforms(train=False)
        ) if train else None

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers
        }
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)

    def __len__(self):
        return len(self.data)

    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset, **self.init_kwargs)


class STLDataLoader(DataLoaderBase):

    def __init__(self, transforms, data_dir, batch_size, shuffle, validation_split, nworkers,
                 train=True):
        self.data_dir = data_dir

        self.train_dataset = datasets.STL10(
            self.data_dir,
            split='train',
            download=True,
            transform=transforms.build_transforms(train=True)
        )
        self.valid_dataset = datasets.STL10(
            self.data_dir,
            split='test',
            download=True,
            transform=transforms.build_transforms(train=False)
        ) if train else None

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers
        }
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)

    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset, **self.init_kwargs)
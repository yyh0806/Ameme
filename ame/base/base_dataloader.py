import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold


class DataLoaderBase(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers=0, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def xfold_split_sampler(self, n_splits):
        idx_full = np.arange(self.n_samples)
        if isinstance(n_splits, int):
            assert n_splits > 0
            assert n_splits < self.n_samples, "validation set size is configured to be larger than entire dataset."
        kf = KFold(n_splits=n_splits)
        res = []
        for train_idx, valid_idx in kf.split(idx_full):
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            train_DataLoader = DataLoader(sampler=train_sampler, **self.init_kwargs)
            valid_DataLoader = DataLoader(sampler=valid_sampler, **self.init_kwargs)
            res.append((train_DataLoader, valid_DataLoader))
        return res

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def get_data(self):
        train_DataLoader = DataLoader(sampler=self.sampler, **self.init_kwargs)
        valid_DataLoader = DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
        return train_DataLoader, valid_DataLoader

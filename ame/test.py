import os
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from albumentations import (HorizontalFlip, VerticalFlip, Normalize, Resize, Compose)
from trainer import Trainer
from metric import *

from .utils import *
from .dataset.CellDataset import *
from .config.cell import *
from .models.UNet import *

fix_all_seeds(2021)

ds_train = CellDataset(df_train, train=True)
dl_train = DataLoader(ds_train, batch_size=2, num_workers=2, pin_memory=True, shuffle=False)
ds_test = CellDataset(df_train, train=False)
dl_test = DataLoader(ds_test, batch_size=1, num_workers=2, pin_memory=True, shuffle=False)


model = UNet(3, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.BCELoss()
metrics = [accuracy]

trainer = Trainer(model, criterion, metrics, optimizer, 20, device, dl_train, dl_test, scheduler)
trainer.train()

if __name__ == "__main__":
    with open(cfg, errors='ignore') as f:
        self.yaml = yaml.safe_load(f)
    cfg.merge_from_file("experiments/config.yml")
    cfg.freeze()
    # print(cfg)
    # train(cfg)
    pass

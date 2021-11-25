import pandas as pd
import cv2

from os.path import join
from torch.utils.data import DataLoader, Dataset

from ame.utils import *


class CellDataset(Dataset):
    def __init__(self, path, transforms, stage="train"):
        self.path = join(path, 'train')
        self.transforms = transforms
        self.stage = stage
        self.df = pd.read_csv(join(path, 'train.csv'))
        self.gb = self.df.groupby('id')
        self.image_ids = np.array(self.df.id.unique())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        df = self.gb.get_group(image_id)
        # Read image
        image_path = os.path.join(self.path, image_id + ".png")
        image = cv2.imread(image_path)
        # Create the mask
        masks = build_masks(df, image_id, input_shape=(520, 704))
        masks = (masks >= 1).astype('float32')
        if self.stage == "train":
            if self.transforms is not None:
                augmented = self.transforms.build_transforms(train=True)(image=image, mask=masks)
                image = augmented["image"]
                masks = augmented["mask"]
                return image, masks.unsqueeze(0)
            return image, masks.unsqueeze(0)
        else:
            if self.transforms is not None:
                augmented = self.transforms.build_transforms(train=False)(image=image, mask=masks)
                image = augmented["image"]
                masks = augmented["mask"]
                return image, masks.unsqueeze(0)
            return image, masks.unsqueeze(0)


# class CellDataset(Dataset):
#     def __init__(self, path, transforms, stage="train"):
#         self.path = join(path, 'train')
#         self.transforms = transforms
#         self.stage = stage
#         self.df = pd.read_csv(join(path, 'train.csv'))
#         self.gb = self.df.groupby('id')
#         self.image_ids = np.array(self.df.id.unique())
#
#     def __len__(self):
#         return len(self.image_ids)
#
#     def __getitem__(self, index):
#         image_id = self.image_ids[index]
#         df = self.gb.get_group(image_id)
#         # Read image
#         image_path = os.path.join(self.path, image_id + ".png")
#         image = cv2.imread(image_path)
#         # Create the mask
#         mask = build_masks(df, image_id, input_shape=(520, 704))
#         mask = (mask >= 1).astype('float32')
#         if self.stage == "train":
#             if self.transforms is not None:
#                 augmented = self.transforms.build_transforms(train=True)(image=image, mask=mask)
#                 image = augmented["image"]
#                 mask = augmented["mask"]
#                 return image, mask.unsqueeze(0)
#             return image, mask.unsqueeze(0)
#         else:
#             if self.transforms is not None:
#                 augmented = self.transforms.build_transforms(train=False)(image=image, mask=mask)
#                 image = augmented["image"]
#                 mask = augmented["mask"]
#                 return image, mask.unsqueeze(0)
#             return image, mask.unsqueeze(0)

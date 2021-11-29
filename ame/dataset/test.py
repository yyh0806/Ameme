import os
import cv2
from torch.utils.data import Dataset
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2


class TestCellDataset(Dataset):
    def __init__(self, test_path):
        self.test_path = test_path

        self.image_ids = [f[:-4] for f in os.listdir(self.test_path)]
        self.num_samples = len(self.image_ids)

        self.transform = Compose([Resize(448, 448),
                                  Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
                                            max_pixel_value=255.0, p=1.0),
                                  ToTensorV2()])

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.test_path, image_id + ".png")
        image = cv2.imread(path)
        image = self.transform(image=image)['image']
        return {'image': image.float(), 'id': image_id}

    def __len__(self):
        return self.num_samples
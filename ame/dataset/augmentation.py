import abc
import cv2
import torchvision.transforms as T
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, Cutout,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, CoarseDropout, ShiftScaleRotate,
    CenterCrop, Resize, ColorJitter
)
from albumentations.pytorch import ToTensorV2


class AugmentationFactoryBase(abc.ABC):
    def build_transforms(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


# -----------------------------------------------------------------------------------------------
#                           CellTransforms
# -----------------------------------------------------------------------------------------------
class CellTransforms(AugmentationFactoryBase):
    def build_train(self):
        train_transform = Compose([
            Resize(512, 512, p=1.0),
            CLAHE(p=0.35),
            ColorJitter(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=90, p=0.5),
            CoarseDropout(max_holes=8, max_height=26, max_width=26, min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)
        ], p=1)
        return train_transform

    def build_test(self):
        test_transform = Compose([
            Resize(512, 512, p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)
        ], p=1)
        return test_transform

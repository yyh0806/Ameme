import abc

import torchvision.transforms as T
import cv2
from Ameme.utils import utils
import numpy as np
from PIL import Image


class AugmentationFactoryBase(abc.ABC):
    def build_transforms(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


class MNISTTransforms(AugmentationFactoryBase):
    MEANS = [0]
    STDS = [1]

    def build_train(self):
        return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])

    def build_test(self):
        return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])


def _preprocess(img, kernel_size=(5, 5)):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    img = np.asarray(img)
    img = clahe.apply(img)

    dst = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, 10)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dst = cv2.erode(dst, kernel1, iterations=2)
    dst = cv2.dilate(dst, kernel2, iterations=2)
    dst = cv2.dilate(dst, kernel3, iterations=3)
    dst = cv2.erode(dst, kernel2, iterations=3)
    dst = Image.fromarray(dst)
    return dst


class SevenSegmentTransforms(AugmentationFactoryBase):

    def build_train(self):
        return T.Compose([T.Grayscale(),
                          T.Lambda(_preprocess),
                          T.Resize((60, 100)),
                          T.ToTensor()])

    def build_test(self):
        return T.Compose([T.Grayscale(),
                          T.Lambda(_preprocess),
                          T.Resize((60, 100)),
                          T.ToTensor()])


class SevenSegmentTransforms2(AugmentationFactoryBase):

    def build_train(self):
        return T.Compose([T.Grayscale(),
                          T.Resize((26, 44)),
                          T.ToTensor()])

    def build_test(self):
        return T.Compose([T.Grayscale(),
                          T.Resize((26, 44)),
                          T.ToTensor()])


class NissinTransforms(AugmentationFactoryBase):

    def build_train(self):
        return T.Compose([T.Resize((60, 100)),
                          T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    def build_test(self):
        return T.Compose([T.Resize((60, 100)),
                          T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])


class STLTransforms(AugmentationFactoryBase):

    def build_train(self):
        return T.Compose([T.Resize((224, 224)),
                          T.RandomHorizontalFlip(),
                          T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    def build_test(self):
        return T.Compose([T.Resize((224, 224)),
                          T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
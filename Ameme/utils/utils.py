import random
import os
import numpy as np
import torch
import cv2


def get_instance(module, node_name, node_params, *args):
    return getattr(module, node_name)(*args, **node_params)


def remove_small_blob(img, min_size):
    """ Remove somall region has pixel less than min_size.

    Note: img must have one channel and be binary, i.e., only two unique values in img.
    """
    # find all connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)

    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2
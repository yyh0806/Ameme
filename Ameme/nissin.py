import os
import random
from typing import Any, List, Tuple, Dict
from types import ModuleType
import torchvision.transforms as T

import numpy as np
import torch
import torch.nn as nn
import torch.optim as module_optimizer
import torch.optim.lr_scheduler as module_scheduler
from PIL import Image
import cv2
import Ameme.data_loader.augmentation as module_aug
import Ameme.data_loader.data_loaders as module_data
import Ameme.model.loss as module_loss
import Ameme.model.metric as module_metric
import Ameme.model.model as module_arch
from Ameme.trainer import Trainer
from Ameme.utils import setup_logger, setup_logging, utils
import yaml

from torchvision.datasets import ImageFolder
from Ameme.utils.CharacterExtraction import CharacterExtraction

log = setup_logger(__name__)


def train(cfg: Dict, resume: str) -> None:
    setup_logging(cfg)
    log.debug(f'Training: {cfg}')
    _seed_everything(cfg['seed'])

    model = get_instance(module_arch, 'arch', cfg)
    model, device = setup_device(model, cfg['target_devices'])
    torch.backends.cudnn.benchmark = True  # disable if not consistent input sizes

    param_groups = setup_param_groups(model, cfg['optimizer'])
    optimizer = get_instance(module_optimizer, 'optimizer', cfg, param_groups)
    lr_scheduler = get_instance(module_scheduler, 'lr_scheduler', cfg, optimizer)
    model, optimizer, start_epoch = resume_checkpoint(resume, model, optimizer, cfg)

    transforms = get_instance(module_aug, 'augmentation', cfg)
    data_loader = get_instance(module_data, 'data_loader', cfg, transforms)
    valid_data_loader = data_loader.split_validation()

    log.info('Getting loss and metric function handles')
    loss = getattr(module_loss, cfg['loss'])
    metrics = [getattr(module_metric, met) for met in cfg['metrics']]

    log.info('Initialising trainer')
    trainer = Trainer(model, loss, metrics, optimizer,
                      start_epoch=start_epoch,
                      config=cfg,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    log.info('Finished!')


def predict(cfg, model_checkpoint, dir_path):
    setup_logging(cfg)
    _seed_everything(config['seed'])

    log.info(f'Using config:\n{config}')

    log.debug('Building model architecture')
    model = get_instance(module_arch, 'arch', cfg)
    model, device = _prepare_device(model, cfg['n_gpu'])

    log.debug(f'Loading checkpoint {model_checkpoint}')
    checkpoint = torch.load(model_checkpoint)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    transforms = T.Compose([T.Resize((60, 100)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

    # prepare model for testing
    model.eval()

    dataset = ImageFolder("D:/Ameme/Ameme/data/nissin")
    dic = dataset.class_to_idx
    inverse_dic = {}
    for key, val in dic.items():
        inverse_dic[val] = key

    with torch.no_grad():
        dir = os.listdir(dir_path)
        for i in range(len(dir)):
            image = cv2.imread("D:/Ameme/data/" + dir[i])
            # Initialize & read image file
            ce = CharacterExtraction("D:/Ameme/data/" + dir[i])

            # Convert to grayscale image
            ce.convert_to_grayscale()

            # Extract area in vertical direction
            cb0, cb1 = ce.extract_character_bands()

            # Calculate shading
            ce.calculate_shading(cb0[1], cb1[0])

            # Extract characters
            coords0 = ce.extract_characters(cb0, 10)  # 10 is dot size of upper character area
            coords1 = ce.extract_characters(cb1, 8)  # 8 is dot size of lower character area

            rois, coords = ce.get_roi(coords0 + coords1)

            for j, img in enumerate(rois):
                pil_img = img
                data = transforms(pil_img)
                data = data.unsqueeze(0).to(device)
                output = model(data)
                output = output.detach().cpu()
                pred = torch.max(output, 1)[1].numpy()
                label = inverse_dic[pred[0]]
                if label == "colon":
                    label = ":"
                elif label == "plus":
                    label = "+"
                elif label == "dot":
                    label = "."
                image = cv2.rectangle(image, (coords[j][0], coords[j][1]), (coords[j][2], coords[j][3]), color=(255, 0, 0), thickness=2)
                image = cv2.putText(image, label, (coords[j][0], coords[j][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 255, 0))

            cv2.imwrite("D:/Ameme/data/" + "result_" + dir[i], image)

    log.info('Finished saving predictions!')


def setup_device(
        model: nn.Module,
        target_devices: List[int]
) -> Tuple[torch.device, List[int]]:
    """
    setup GPU device if available, move model into configured device
    """
    available_devices = list(range(torch.cuda.device_count()))

    if not available_devices:
        log.warning(
            "There's no GPU available on this machine. Training will be performed on CPU.")
        device = torch.device('cpu')
        model = model.to(device)
        return model, device

    if not target_devices:
        log.info("No GPU selected. Training will be performed on CPU.")
        device = torch.device('cpu')
        model = model.to(device)
        return model, device

    max_target_gpu = max(target_devices)
    max_available_gpu = max(available_devices)

    if max_target_gpu > max_available_gpu:
        msg = (f"Configuration requests GPU #{max_target_gpu} but only {max_available_gpu} "
               "available. Check the configuration and try again.")
        log.critical(msg)
        raise Exception(msg)

    log.info(f'Using devices {target_devices} of available devices {available_devices}')
    device = torch.device(f'cuda:{target_devices[0]}')
    if len(target_devices) > 1:
        model = nn.DataParallel(model, device_ids=target_devices).to(device)
    else:
        model = model.to(device)
    return model, device


def setup_param_groups(model: nn.Module, config: Dict) -> List:
    return [{'params': model.parameters(), **config}]


def resume_checkpoint(resume_path, model, optimizer, config):
    """
    Resume from saved checkpoint.
    """
    if not resume_path:
        return model, optimizer, 0

    log.info(f'Loading checkpoint: {resume_path}')
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    if checkpoint['config']['optimizer']['type'] != config['optimizer']['type']:
        log.warning("Warning: Optimizer type given in config file is different from "
                    "that of checkpoint. Optimizer parameters not being resumed.")
    else:
        optimizer.load_state_dict(checkpoint['optimizer'])

    log.info(f'Checkpoint "{resume_path}" loaded')
    return model, optimizer, checkpoint['epoch']


def get_instance(
        module: ModuleType,
        name: str,
        config: Dict,
        *args: Any
) -> Any:
    """
    Helper to construct an instance of a class.

    Parameters
    ----------
    module : ModuleType
        Module containing the class to construct.
    name : str
        Name of class, as would be returned by ``.__class__.__name__``.
    config : dict
        Dictionary containing an 'args' item, which will be used as ``kwargs`` to construct the
        class instance.
    args : Any
        Positional arguments to be given before ``kwargs`` in ``config``.
    """
    ctor_name = config[name]['type']
    log.info(f'Building: {module.__name__}.{ctor_name}')
    return getattr(module, ctor_name)(*args, **config[name]['args'])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _prepare_device(model, n_gpu_use):
    device, device_ids = _get_device(n_gpu_use)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model, device


def _get_device(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        log.warning("Warning: There\'s no GPU available on this machine,"
                    "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        log.warning(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, "
                    f"but only {n_gpu} are available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    log.info(f'Using device: {device}, {list_ids}')
    return device, list_ids


if __name__ == "__main__":
    with open("D:/Ameme/experiments/config_nissin.yml") as fh:
        config = yaml.safe_load(fh)
    # train(config, None)
    predict(config, "D:/Ameme/Ameme/saved/Nissin/0105-131717/checkpoints/model_best.pth", "D:/Ameme/data")

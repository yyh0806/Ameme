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
    transforms = T.Compose([T.Grayscale(), T.Resize((60, 100)), T.ToTensor(), T.Normalize([0], [1])])

    # prepare model for testing
    model.eval()

    with torch.no_grad():
        dir = os.listdir(dir_path)
        for i in range(len(dir)):
            img = cv2.imread("D:/Ameme/data/" + dir[i])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = preprocess(gray, 5)
            pil_img = Image.fromarray(gray)
            data = transforms(pil_img)
            data = data.unsqueeze(0).to(device)
            output = model(data)
            output = output.detach().cpu()
            pred = torch.max(output, 1)[1].numpy()
            print(pred)

    log.info('Finished saving predictions!')


def predict_movie(cfg, model_checkpoint, movie_path, output_path):
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
    # transforms = T.Compose([T.Grayscale(), T.Resize((60, 100)), T.ToTensor()])
    transforms = get_instance(module_aug, 'augmentation', cfg).build_test()
    # prepare model for testing
    model.eval()

    with torch.no_grad():
        cap = cv2.VideoCapture(movie_path)
        ret, frame = cap.read()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]), True)
        while ret:
            cv2.rectangle(frame, (20, 220), (240, 420), (0, 0, 255), 3)
            cv2.rectangle(frame, (600, 380), (1540, 800), (255, 0, 0), 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mat_condition = gray[220:420, 20:240]
            pred_1 = gray[380:800, 620:840]
            pred_2 = gray[380:800, 840:1060]
            pred_3 = gray[380:800, 1060:1280]
            pred_4 = gray[380:800, 1280:1500]
            pred1 = " "
            pred2 = " "
            pred3 = " "
            pred4 = " "
            threshold = 6000
            test1 = cv2.adaptiveThreshold(
                pred_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, 10)
            test2 = cv2.adaptiveThreshold(
                pred_2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, 10)
            test3 = cv2.adaptiveThreshold(
                pred_3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, 10)
            test4 = cv2.adaptiveThreshold(
                pred_4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, 10)

            if cv2.sumElems(mat_condition)[0] < 5900000:
                if np.sum(test1[test1 == 255]) / 255 > threshold:
                    pil_img1 = Image.fromarray(pred_1)
                    data1 = transforms(pil_img1)
                    data1 = data1.unsqueeze(0).to(device)
                    output1 = model(data1)
                    output1 = output1.detach().cpu()
                    pred1 = torch.max(output1, 1)[1].numpy()[0]
                if np.sum(test2[test2 == 255]) / 255 > threshold:
                    pil_img2 = Image.fromarray(pred_2)
                    data2 = transforms(pil_img2)
                    data2 = data2.unsqueeze(0).to(device)
                    output2 = model(data2)
                    output2 = output2.detach().cpu()
                    pred2 = torch.max(output2, 1)[1].numpy()[0]
                if np.sum(test3[test3 == 255]) / 255 > threshold:
                    pil_img3 = Image.fromarray(pred_3)
                    data3 = transforms(pil_img3)
                    data3 = data3.unsqueeze(0).to(device)
                    output3 = model(data3)
                    output3 = output3.detach().cpu()
                    pred3 = torch.max(output3, 1)[1].numpy()[0]
                if np.sum(test4[test4 == 255]) / 255 > threshold:
                    pil_img4 = Image.fromarray(pred_4)
                    data4 = transforms(pil_img4)
                    data4 = data4.unsqueeze(0).to(device)
                    output4 = model(data4)
                    output4 = output4.detach().cpu()
                    pred4 = torch.max(output4, 1)[1].numpy()[0]
                cv2.putText(frame, str(pred1) + str(pred2) + str(pred3) + str(pred4), (600, 380),
                            cv2.FONT_HERSHEY_PLAIN, 8.0, (0, 255, 0), 2)
            #cv2.imshow("test", frame)
            #cv2.waitKey(10)
            out.write(frame)
            ret, frame = cap.read()

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


def _preprocess(img, kernel_size=(5, 5)):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    img = clahe.apply(img)

    dst = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, 10)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dst = cv2.erode(dst, kernel1, iterations=2)
    dst = cv2.dilate(dst, kernel2, iterations=2)
    dst = cv2.dilate(dst, kernel3, iterations=3)
    dst = cv2.erode(dst, kernel2, iterations=3)
    # dst = utils.remove_small_blob(dst, 300)
    dst = 255 - dst
    # cv2.imshow("dst", dst)
    # cv2.waitKey(1000)
    return dst


if __name__ == "__main__":
    with open("D:/Ameme/experiments/config.yml") as fh:
        config = yaml.safe_load(fh)
    # train(config, None)
    # predict(config, "D:/Ameme/Ameme/saved/SevenSegment/1225-163859/checkpoints/model_best.pth", "D:/Ameme/data")

    predict_movie(config, "D:/Ameme/Ameme/saved/SevenSegment/1228-111729/checkpoints/model_best.pth",
                  "D:/seven-segment-ocr/data/WIN_20201215_17_53_40_Pro.mp4", "D:/20201228/7seg/test.mp4")

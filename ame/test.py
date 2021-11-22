from trainer import Trainer
from metric import *
from loguru import logger
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.nn import *

from ame.utils import *
from ame.dataset.dataloaders import *
from ame.models.UNet import *
from config import *

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("experiments/cell.yml")
    cfg.freeze()
    logger.info(cfg)
    # seed
    fix_all_seeds(cfg['SEED'])
    # dataloader
    dataloader = eval(cfg["DATALOADER"]["TYPE"])(**cfg["DATALOADER"]["ARGS"])
    valid_dataloader = dataloader.split_validation()
    # build model architecture
    model = eval(cfg["MODEL"]["TYPE"])(**cfg["MODEL"]["ARGS"])
    logger.info(model)
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg['N_GPU'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # optimizer and scheduler
    optimizer = eval(cfg["OPTIMIZER"]["TYPE"])(**cfg["OPTIMIZER"]["ARGS"], params=model.parameters())
    scheduler = eval(cfg["SCHEDULER"]["TYPE"])(**cfg["SCHEDULER"]["ARGS"], optimizer=optimizer)
    # get function handles of loss and metrics
    criterion = eval(cfg["LOSS"])
    metrics = [eval(met) for met in cfg["METRICS"]]
    logger.info("train")
    trainer = Trainer(model, criterion, metrics, optimizer, cfg["EPOCH"],
                      device, dataloader, valid_dataloader, scheduler)
    trainer.train()

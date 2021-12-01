from trainer import Trainer
from metric import *
from loguru import logger
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.nn import *

from ame.utils import *
from ame.dataset.dataloaders import *


from config import *

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("experiments/test_FPN.yml")
    cfg.freeze()
    logger.info(cfg)
    # seed
    fix_all_seeds(cfg['SEED'])
    # dataloader
    dataloader = eval(cfg["DATALOADER"]["TYPE"])(**cfg["DATALOADER"]["ARGS"])
    valid_dataloader = dataloader.split_validation()
    # build model architecture
    model = None
    # --------------------------------- segmentation ---------------------------------
    encoder = None
    decoder = None
    segmentation_head = None
    classification_head = None
    if cfg["MODEL"]["TYPE"] == "segmentation":
        from segmentation.encoders import Encoder
        from segmentation.decoders import Decoder
        from segmentation.model import SegModel
        from segmentation.base import SegmentationHead, ClassificationHead
        encoder = Encoder(**cfg["MODEL"]["ARGS"]["Encoder"])
        decoder = Decoder(**cfg["MODEL"]["ARGS"]["Decoder"])
        segmentation_head = SegmentationHead(**cfg["MODEL"]["ARGS"]["Segmentation_head"])
        if "Classification_head" in cfg["MODEL"]["ARGS"]:
            classification_head = ClassificationHead(**cfg["MODEL"]["ARGS"]["Classification_head"])
        assert encoder is not None
        assert decoder is not None
        assert segmentation_head is not None
        model = SegModel(encoder, decoder, segmentation_head, classification_head)
    # --------------------------------------------------------------------------------
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
    trainer = Trainer(model, criterion, metrics, optimizer, cfg["EPOCH"],
                      device, dataloader, valid_dataloader, scheduler,
                      checkpoint=cfg["CHECKPOINT"], save_dir=cfg["SAVE_DIR"])
    trainer.train()

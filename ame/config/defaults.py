from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)
_C.NAME = "Ameme"
_C.EPOCH = 20
_C.SAVE_DIR = "saved/"
_C.SEED = 8060
_C.N_GPU = 1
_C.RESUME = None

_C.DATA_LOADER = CN(new_allowed=True)
_C.DATA_LOADER.TYPE = ""
_C.DATA_LOADER.ARGS = CN(new_allowed=True)

_C.MODEL = CN(new_allowed=True)
_C.MODEL.TYPE = ""
_C.MODEL.ARGS = CN(new_allowed=True)

_C.OPTIMIZER = CN(new_allowed=True)
_C.OPTIMIZER.TYPE = ""
_C.OPTIMIZER.ARGS = CN(new_allowed=True)

_C.SCHEDULER = CN(new_allowed=True)
_C.SCHEDULER.TYPE = ""
_C.SCHEDULER.ARGS = CN(new_allowed=True)

_C.LOSS = ""

_C.METRICS = ["top_1_acc"]


def get_cfg_defaults():
    return _C.clone()

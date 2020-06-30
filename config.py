from yacs.config import CfgNode as CN
import math
_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.USE_CUDA = False

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.CSV = 'train_split.csv'
_C.TRAIN.DATA_DIR = 'train'
_C.TRAIN.NUM_WORKERS = 0

_C.EVALUATE = CN()
_C.EVALUATE.BATCH_SIZE = 32
_C.EVALUATE.CSV = 'valid_split.csv'
_C.EVALUATE.DATA_DIR = 'train'
_C.EVALUATE.PREDICTION_CSV = 'valid_prediction.csv'
_C.EVALUATE.NUM_WORKERS = 8

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 32
_C.TEST.CSV = 'test.csv'
_C.TEST.DATA_DIR = 'test'
_C.TEST.PREDICTION_CSV = 'test_prediction.csv'
_C.TEST.NUM_WORKERS = 8

_C.MODEL = CN()
_C.MODEL.NAME = 'ResNext50_32x4d'
_C.MODEL.CHECKPOINT = ""

_C.OPTIM = CN()
_C.OPTIM.INIT_LR = 0.001

_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.STEP_SIZE = 120

_C.AUGMENT = CN()
_C.AUGMENT.RESIZE = (224, 224)
_C.AUGMENT.NORMALIZE = False
_C.AUGMENT.FLIP = False
_C.AUGMENT.FLIP_PROB = 0.5

_C.EPOCH = 200
_C.EVAL_MODEL_EVERY_EPOCH = 10
_C.TAG = ''

def get_cfg_defaults():
    return _C.clone()
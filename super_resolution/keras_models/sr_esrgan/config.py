from typing import Union

from tensorflow.python.keras.backend_config import _EPSILON

####################EXPERIMENTS#######################

CONV_NUM: int = 2 # in linear part after upscale
RES_NUM: int = 8
INPUT_SHAPE: Union[None, int]  = (None, None, 3)
EPOCHS: int = 100
BATCH_SIZE: int = 32
BATCHNORM: bool = False

MEAN: list = [value * 255 for value in [0.4488, 0.4371, 0.4040]]
STD: float = 127.5

SPE: int = 600 # steps per epochs
VS: int = 10 # validations steps
PERIOD: int = 10 # save each

# generator features
FINAL_SHORTCUT: bool = True
UPSAMPLING_BLOCKS: int = 2
FINAL_ACTIVATIONQ: bool = False

# discriminator features
MOMENTUM: float = 0.5 # for batchnorm
ALPHA: float = 0.2

FEATURES_MAP: dict = {
    'level_1': [64, 3, 2],
    'level_2': [128, 3, 1],
    'level_3': [128, 3, 2],
    'level_4': [256, 3, 1],
    'level_5': [256, 3, 2],
    'level_6': [512, 3, 1],
    'level_7': [512, 3, 2]}

DENSE_NEURONS: int = 1024
FINAL_ACTIVATION: str = 'sigmoid'

#######################loss############################
TOP: bool = False
TRAINABLE: bool = False
WHEIGHTS: str = 'imagenet'
OUT_LAYER: str = 'block5_pool'


#######################LEARNING_RATES###################

BOUNDARIES: list = [1000]
VALUES: list = [1e-4, 7e-5]
LAMBDA_PARAM: float = 0.1
LOSS_WEIGHTS: list = [1., 1e-3]
LEARNING_RATE: float = 1e4
BETA_1: float = 0.9
BETA_2: float = 0.999
EPSILON: float = 1e-08

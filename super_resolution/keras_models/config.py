from typing import Union

####################EXPERIMENTS#######################

CONV_NUM: int = 2 # in linear part after upscale
RES_NUM: int = 16
INPUT_SHAPE: Union[None, int]  = (None, None, 3)
LEARNING_RATE: float = 0.00001
EPOCHS: int = 100
BATCH_SIZE: int = 32
BATCHNORM: bool = False

MEAN: list = [value * 255 for value in [0.4488, 0.4371, 0.4040]]
STD: float = 127.5

SPE: int = 600 # steps per epochs
VS: int = 10 # validations steps
PERIOD: int = 10 # save each 

#######################LEARNING_RATES###################

BOUNDARIES: list = [1000]
VALUES: list = [1e-4, 7e-5]

LAMBDA_PARAM: float = 0.1

TOP: bool = False
TRAINABLE: bool = False
WHEIGHTS: str = 'imagenet'
# OUT_LAYER: str = 'block5_pool'
OUT_LAYER: str = 'block5_conv4'
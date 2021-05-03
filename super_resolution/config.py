from typing import Union

####################EXPERIMENTS#######################

CONV_NUM: int = 2
RES_NUM: int = 6
INPUT_SHAPE: Union[None, int]  = (None, None, 3)
LEARNING_RATE: float = 0.00001
EPOCHS: int = 100
BATCH_SIZE: int = 64
BATCHNORM: bool = False

MEAN: list = [value * 255 for value in [0.4488, 0.4371, 0.4040]]
STD: float = 127.5

SPE: int = 1000
VS: int = 10
PERIOD: int = 10

#######################LEARNING_RATES###################

BOUNDARIES: list = [3000]
VALUES: list = [1e-4, 5e-5]
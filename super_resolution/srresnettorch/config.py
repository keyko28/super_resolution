import numpy as np

LERANING_RATE: float = 0.0001 # 0.000005
MAX_LR: float = 0.00005
B1: float = 0.5
B2: float = 0.999
SHAPES: list = [256, 256]

# next is from the calc_stat module
MEAN: np.ndarray = np.array([0.45328920380047055, 0.43493361995304664, 0.4093699853405604])
STD: np.ndarray = np.array([0.2487343734737629, 0.23233310286204029, 0.22754716138014985])
DOWNSAMPLE_FACTOR: int = 2
EPOCHS: int =  40
BATCH_SIZE: int  = 4
SAMPLE_INTERVAL: int = 100
BATCH_SAVE_PATH: str = 'D:\\pet_projects\\super_resolution\\srresnettorch\\test_images'
MODEL_SAVE_PATH: str = 'D:\\pet_projects\\super_resolution\\srresnettorch\\model_weights'
CHECKPOINT: int = 10
DS_PATH: str = 'D:\\pet_projects\\super_resolution\\dataset'
LR_UP: int = 2000
GAMMA: float = 0.1
STEP_SIZE: int = 20


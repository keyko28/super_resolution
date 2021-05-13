from generator import Generator
import config as cfg
import numpy as np
from PIL import Image
import glob
from typing import List
import sys
from keras.layers import AveragePooling2D


def normalize(x):

        return (x-cfg.MEAN)/cfg.STD
        

def denormalize(x):

        return [x * cfg.STD + value for value in cfg.MEAN]


def load_train_images_names(lr_train_path: str,
                  hr_train_path: str,
                  lr_test_path: str,
                  hr_test_path: str,
                  extension: str):

    lr_train_images: list = glob.glob(f"{lr_train_path}" + '\\' + f"*.{extension}")
    hr_train_images: list = glob.glob(f"{hr_train_path}" + '\\' + f"*.{extension}")
    lr_test_images: list = glob.glob(f"{lr_test_path}" + '\\' + f"*.{extension}")
    hr_test_images: list = glob.glob(f"{hr_test_path}" + '\\' + f"*.{extension}")

    images = [lr_train_images, hr_train_images, lr_test_images, hr_test_images]

    return images

def get_data(images: List[str]) -> object:

        return map(lambda x: normalize(np.array(Image.open(x))), images)


def stdout_write(message: List[str]) -> None:

        for line in message:
                sys.stdout.write(line)


def generator_predict(generator_model: Generator, images: List[np.ndarray]):
        
        predictions: list = []

        for image in images:

                image = np.expand_dims(image, axis=0)
                image = AveragePooling2D(2)(image)

                predictions.append(generator_model.predict(image))

        return predictions




        





# p1 = 'D:\\pet_projects\\super_resolution\\.div2k\\images\\DIV2K_train_LR_unknown\X4'
# p2 = 'D:\\pet_projects\\super_resolution\\.div2k\\images\\DIV2K_train_HR\X4'
# p3 = 'D:\\pet_projects\\super_resolution\\.div2k\\images\\DIV2K_valid_LR_unknown\X4'
# p4 = 'D:\\pet_projects\\super_resolution\\.div2k\\images\\DIV2K_valid_HR\X4'

# print(load_train_df(p1, p2, p3, p4, 'png'))

    

    

    


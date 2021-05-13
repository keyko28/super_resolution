import build_model as bm
import config as cfg
import numpy as np
import tensorflow as tf
from keras.models import load_model
from typing import Callable
from utils import plot_sample, load_image


def resolve(low_res, model: Callable):

    lr = tf.expand_dims(low_res, axis=0)
    lr_batch = tf.cast(lr, tf.float32)

    sr_batch = model.predict(lr_batch)

    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)

    return sr_batch[0]

def get_model(model_name: str = 'models/',
              weights_path: str = 'D:\\pet_projects\\super_resolution\\models\\mixGE\\weights',
              weightsQ: bool = True) -> bm.ModelBuilder:

    if weightsQ:

        model = bm.ModelBuilder(cfg.MEAN, cfg.STD)
        model.compile()
        model.load_wheights(weights_path)

        return model

    else:

        model = bm.ModelBuilder(cfg.MEAN, cfg.STD)
        model.compile()

        return load_model(model_name, custom_objects={'normalize': model._normalize, 'denormalize': model._denormalize})  


def demo_plot(low_res_path: str,
              model_name: str = 'models\\super_resolution_leaky_relu',
              weights_path: str = 'weights/sr_model_epoch100.hdf5',
              weightsQ: bool = True) -> None:

    # model = bm.ModelBuilder(cfg.MEAN, cfg.STD)
    # model.compile()
    # model.load_wheights(weights_path)

    #TODO CORRECT SAVE MODEL

    model = get_model(weightsQ=True, weights_path=weights_path)

    lr = load_image(low_res_path)
    sr = resolve(lr, model)

    plot_sample(lr, sr, saveQ=False)


def main():

    lr_path = 'D:\\pet_projects\\super_resolution\\.div2k\\images\\DIV2K_train_LR_unknown\\X4\\0012x4.png'

    demo_plot(lr_path)



if __name__ == '__main__':
    
    main()
    

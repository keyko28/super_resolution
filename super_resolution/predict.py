import build_model as bm
import config as cfg
import numpy as np
import tensorflow as tf
from typing import Callable
from utils import plot_sample, load_image


def resolve(low_res, model: Callable):
    lr = tf.expand_dims(low_res, axis=0)
    lr_batch = tf.cast(lr, tf.float32)
    sr_batch = model.predict(lr_batch)
    # sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch[0]


def demo_plot(low_res_path: str, weights_path: str = 'weights/sr_model_epoch01.hdf5' ) -> None:

    model = bm.ModelBuilder(cfg.MEAN, cfg.STD)
    model.compile()
    model.load_wheights(weights_path)    

    lr = load_image(low_res_path)
    sr = resolve(lr, model)

    plot_sample(lr, sr, saveQ=False)


def main():

    lr_path = 'D:\\pet_projects\\super_resolution\\.div2k\\images\\DIV2K_train_LR_unknown\\X4\\0001x4.png'

    demo_plot(lr_path)



if __name__ == '__main__':
    
    main()
    

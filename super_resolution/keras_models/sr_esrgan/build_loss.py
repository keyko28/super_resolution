from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
import keras.backend as K
from keras.models import Model
import config as cfg
import numpy as np
from PIL import Image
from keras.layers import AveragePooling2D
from typing import Tuple, Union
import tensorflow as tf


class Loss:
    
    def __init__(self):
        self.shape = cfg.INPUT_SHAPE

        #from config
        self.include_top = cfg.TOP
        self.trainable = cfg.TRAINABLE
        self.weights = cfg.WHEIGHTS
        self.output_layer = cfg.OUT_LAYER


    def _preprocess_image(self, 
                          y_true: np.ndarray , 
                          y_pred: np.ndarray, 
                          factor: Union[int, Tuple] = 2)->Tuple[tf.Tensor]:

        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        if len(y_true.shape) == 3 and len(y_pred.shape) == 3:
        
            # y_true = np.expand_dims(y_true, axis=0)
            # y_pred = np.expand_dims(y_pred, axis=0)

            y_true = tf.expand_dims(y_true, axis=0, name=None)
            y_pred = tf.expand_dims(y_pred, axis=0, name=None)

        # y_true = preprocess_input(y_true)
        # y_pred = preprocess_input(y_pred)

        y_true = AveragePooling2D(pool_size=factor)(y_true)
        y_pred = AveragePooling2D(pool_size=factor)(y_pred)



        return y_true, y_pred

    def _compile_loss(self, y_true, y_pred):

        # tf.config.run_functions_eagerly(True)

        y_true, y_pred = self._preprocess_image(y_true, y_pred, factor=6)

        network = VGG19(include_top=self.include_top, weights=self.weights, input_shape=self.shape)
        network.trainable = self.trainable

        for layer in network.layers:
            layer.trainable = self.trainable

        model = Model(inputs = network.input, outputs=network.get_layer(self.output_layer).output)
        model.trainable = self.trainable

        # y_true = model.predict(y_true)
        # y_pred = model.predict(y_pred)

        y_true = model(y_true)
        y_pred = model(y_pred)

        return K.mean(K.square(y_true - y_pred))



# l = Loss()
# y_true = np.array(Image.open('D:\\Photo\\09052021\\jpgs\\DSC_2052.jpg'))


# y_pred = np.array(Image.open('D:\\Photo\\09052021\\jpgs\\DSC_2052-2.jpg'))
# print(l._compile_loss(y_true, y_pred))
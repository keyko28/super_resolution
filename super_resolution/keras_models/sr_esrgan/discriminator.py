import config as cfg
from keras.layers import Dense, Input
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D
from keras.layers import ReLU, LeakyReLU, PReLU
from keras.layers.core import Flatten
from keras.models import Model
from keras.layers import Reshape
from typing import List, Union

class Discriminator:

    def __init__(self):

        self.input_shape = cfg.INPUT_SHAPE
        self.batchQ = cfg.BATCHNORM
        self.momentum = cfg.MOMENTUM # in batch
        self.alpha = cfg.ALPHA # in leakyrelu
        self.features_map = cfg.FEATURES_MAP # in build
        self.dense_neurons = cfg.DENSE_NEURONS
        self.final_activation = cfg.FINAL_ACTIVATION

        self.model = None
        

    def _add_conv(self, x, name:str,\
                  filters: int=64,\
                  size: Union[tuple, int] = 3,\
                  strides: Union[tuple, int] = (1,1),\
                  padding: str = 'same'):

        return Conv2D(filters = filters, kernel_size = size, strides = strides, padding = padding, name=name)(x)


    def _add_dicriminator_block(self, name: str, x, filters: int, size: Union[tuple, int],  strides: Union[tuple, int] = 1, padding: str = 'same'):

        X = self._add_conv(x=x, name=name, size=size, filters=filters, strides=strides)

        if self.batchQ:

            X = BatchNormalization(momentum=self.momentum)(X)

        X = LeakyReLU(alpha=self.alpha)(X)

        return X


    def _build_discriminator(self):

        discr_input = Input(shape=self.input_shape)

        X = self._add_conv(x=discr_input, name='conv_layer_0', filters=64, size=3, strides=1)
        X = LeakyReLU(alpha=self.alpha)(X)

        for key, value in self.features_map.items():

            filters, size, strides = value
            name = 'discriminator_block_' + str(key)

            X = self._add_dicriminator_block(x=X, name=name, filters=filters, size=size, strides=strides)

        needed_shape = X.shape[-1]

        X = Flatten()(X)
        X = Reshape((-1, needed_shape))(X)
        X = Dense(self.dense_neurons)(X)
        X = LeakyReLU(alpha = self.alpha)(X)
        X = Dense(1)(X)

        X = Activation(self.final_activation)(X)

        self.model =  Model(inputs = discr_input, outputs = X)

        return self.model

    def _summary(self):

        self.model.summary()


# d = Discriminator()
# d._build_discriminator()
# d._summary()






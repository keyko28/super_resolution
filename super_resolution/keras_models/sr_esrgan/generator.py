import config as cfg
from keras.layers import Dense, Input, Add
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2DTranspose, Conv2D
from keras.layers import ReLU, LeakyReLU, PReLU
from keras.layers.core import Flatten
from keras.models import Model
from typing import Callable, List, Union

class Generator:

    shortcuts: list = []
    
    def __init__(self, mean_df: List[int], std_df: float, activation_layer: Callable = ReLU):

        self.mean_df = mean_df
        self.std_df = std_df
        self.activation_layer = activation_layer

        #from config
        self.batchQ = cfg.BATCHNORM
        self.input_shape = cfg.INPUT_SHAPE
        self.res_block_num = cfg.RES_NUM
        self.final_shortcut = cfg.FINAL_SHORTCUT
        self.upsampling_blocks = cfg.UPSAMPLING_BLOCKS
        self.final_activation = cfg.FINAL_ACTIVATIONQ

        self.model = None


    def _build_gen(self):

        gen_input = Input(shape = self.input_shape)
        X = self._add_conv(x = gen_input, filters=64, size=9, strides=1, padding='same', name='layer_0_conv')
        # X = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(X)
        X = LeakyReLU()(X)

        main_shortcut = X

        for i in range(self.res_block_num):

            X = self._add_resblock(X, block_num=i)

        X = self._add_conv(x = X, filters=64, size=3, strides=1, padding='same', name='after_res_conv')

        if self.batchQ:

            X = BatchNormalization(name=f'BatchNorm_after_res')(X)

        if self.final_shortcut:

            X = Add()([main_shortcut, X, *Generator.shortcuts])

        else:

            X = Add()([main_shortcut, X])

        for i in range(self.upsampling_blocks):

            X = self._add_upsampling_block(x= X, num=i, filters=256, size=3, padding='same', strides=1)

        X = self._add_conv(x = X, filters=3, size = 9, strides=1, name = 'final_conv')

        if self.final_activation:
            X = Activation('tanh')(X)

        self.model = Model(inputs = gen_input, outputs = X)

        return self.model

        
    def _add_conv(self, x, name:str,\
                  filters: int=64,\
                  size: Union[tuple, int] = 3,\
                  strides: Union[tuple, int] = (1,1),\
                  padding: str = 'same'):

        return Conv2D(filters = filters, kernel_size = size, strides = strides, padding = padding, name=name)(x)

    
    def _add_resblock(self, x, block_num: int):

        X_shortcut = x
        Generator.shortcuts.append(X_shortcut) # add variable to class attr

        X = self._add_conv(x, strides = (1, 1), name=f'ConvLayer_{block_num}')

        if self.batchQ: # using batch norm esrgan could be transformed to srgan

            X = BatchNormalization(name=f'BatchNorm_{block_num}')(X)

        X = self.activation_layer(name = f'ActivationLayer_{block_num}')(X)
        X = self._add_conv(x, strides = (1, 1), name=f'ConvLayer_{block_num}')

        if self.batchQ:

            X = BatchNormalization(name=f'BatchNorm_{block_num}')(X)
        
        X = Add()([X, *Generator.shortcuts])

        return X


    def _add_upscale(self, name:str,\
                  filters: int=48, # 48
                  size: tuple = 4, 
                  strides: tuple = (4,4),
                  padding: str = 'same'):

        return Conv2DTranspose(filters, size, strides, name=name, padding=padding)


    def _add_upsampling_block(self, x, num, filters, size, strides, padding):

        X = self._add_conv(x, name = f'Upsample_Part_Conv_Layer_{num}', filters=filters)
        X = self._add_upscale(name=f'UPSACLE_LAYER_{num}')(X)
        # X = PReLU()(X)
        X = LeakyReLU()(X)

        return X

    def _summary(self):

        self.model.summary()

    
m = Generator(mean_df = cfg.MEAN, std_df = cfg.STD)
m._build_gen()
m._summary()



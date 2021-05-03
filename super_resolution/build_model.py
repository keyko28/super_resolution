# res net16
from keras.layers import Conv2D, Input, Add, Lambda
from keras.layers import Conv2DTranspose, BatchNormalization, ReLU
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import config as cfg
from typing import Callable
import numpy as np
from keras.optimizers.schedules import PiecewiseConstantDecay



class ModelBuilder:

    def __init__(self, mean_df, std_df):

        self.mean_df = mean_df
        self.std_df = std_df

        self.model = None


    def compile(self, optimizer: Callable = Adam, lr: float = cfg.LEARNING_RATE):

        self.model = self._build_model()
        opt = optimizer(self._learning_rate())

        self.model.compile(optimizer=opt, loss='mae', metrics=['accuracy'])

    def model_train(self, x_train = None, y_train = None):

        callbacks = self._create_check_point()

        # compile first
        self.model.fit(x_train, validation_data = y_train, epochs=cfg.EPOCHS, 
                        steps_per_epoch=cfg.SPE, validation_steps=cfg.VS,
                        callbacks=callbacks)


    def load_wheights(self, path:str) -> None:

        self.model.load_weights(path)  


    def _create_check_point(self):

        filepath = 'weights/sr_model_epoch{epoch:02d}.hdf5'
        checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min',
                             save_freq=int(cfg.PERIOD * cfg.SPE))

        return [checkpoint]

        
    def _learning_rate(self):

        return PiecewiseConstantDecay(boundaries=cfg.BOUNDARIES, values=cfg.VALUES)

    def _summary(self):

        self.model.summary()

    def _eval(self, is_verbose: bool = True, x_test = None, y_test = None):

        preds = self.model.evaluate(x_test, y_test, batch_size=cfg.BATCH_SIZE, verbose=is_verbose, sample_weight=None)

        if is_verbose:
            
            self.model.summary()
            print()
            print ("Loss = " + str(preds[0]))
            print ("Test Accuracy = " + str(preds[1]))
            

    def model_save(self, path: str = "./models"):

        self.model.save(path)


    def _normalize(self, x):

        return (x-self.mean_df)/self.std_df


    def _denormalize(self, x):

        return [x * self.std_df + value for value in self.mean_df]


    def _add_conv(self, x, name:str,\
                  filters: int=32,\
                  size: tuple = 3,\
                  strides: tuple = (1,1),\
                  padding: str = 'same'):

        return Conv2D(filters = filters, kernel_size = size, strides = strides, padding = padding, name=name)(x)


    def _add_resblock(self, x, block_num: int, batchQ: bool = cfg.BATCHNORM, activation_layer: Callable = ReLU):
        X_shortcut = x
        X = self._add_conv(x, strides = (1, 1), name=f'ConvLayer_{block_num}')

        if batchQ:

            X = BatchNormalization(name=f'BatchNorm_{block_num}')(X)

        X = activation_layer(name = f'ActivationLayer_{block_num}')(X)
        X = Add()([X, X_shortcut])

        return X


    def _non_linear_part(self, x):

        X = Lambda(self._normalize)(x)
        X = self._add_conv(X, name=f'Convolution_layer_0')
        through_shortcut = X

        for i in range(cfg.RES_NUM):

            X = self._add_resblock(x = X, block_num = i)

        X = Add()([X, through_shortcut])

        return X


    def _add_upscale(self, name:str,\
                  filters: int=48,\
                  size: tuple = 4,\
                  strides: tuple = (4,4),
                  padding: str = 'same'):

        return Conv2DTranspose(filters, size, strides, name=name, padding=padding)


    def _linear_part(self, x):
        
        X = self._add_upscale(name = f'UPSACLE_LAYER_0')(x)

        for i in range(cfg.CONV_NUM):

            X = self._add_conv(X, name = f'Linear_Part_Conv_Layer_{i}', filters=3)

        X = Lambda(self._denormalize)(X)

        return X


    def _build_model(self):

        X_input = Input(cfg.INPUT_SHAPE)
        X = self._non_linear_part(x = X_input)
        X = self._linear_part(X)

        model = Model(inputs=X_input, outputs=X, name='super_resolution')

        return model


    def _load_model(self, model_name: str):
        """
        cd and model location must be equal
        """

        return load_model(model_name)

    def predict(self, data):

        return self.model.predict(data)

    #TODO
    """
    watch over archictecture
    UPSAMPLE +
    PIC_SHUFFLE??
    SUMMARY + 
    COMPILE + 
    TRAIN +
    MODEL SAVE+
    MODEL LOAD+
    GET_UPSACLE_PIC+
    LOSS??
    DATA LOADER
    ESRGAN
    """

        






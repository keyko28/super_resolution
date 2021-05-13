# res net16
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.losses import mean_squared_error
from keras.layers import Conv2D, Input, Add, Lambda
from keras.layers import Conv2DTranspose, BatchNormalization, ReLU, LeakyReLU
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import config as cfg
from typing import Callable, List, Union
import numpy as np
from keras.optimizers.schedules import PiecewiseConstantDecay



class ModelBuilder:

    def __init__(self, mean_df: List[int], std_df: float, activation_layer: Callable = ReLU):

        self.mean_df = mean_df
        self.std_df = std_df
        self.activation_layer = activation_layer

        self.model = None


    def compile(self, 
                optimizer: Callable = Adam,
                loss: Union[Callable, str] = 'mae'):

        self.model = self._build_model()
        opt = optimizer(self._learning_rate())

        self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

        # if loss != 'custom':

        #     self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

        # else:
        #     self.model.compile(optimizer=opt, metrics=['accuracy'])


    def model_train(self, x_train = None, y_train = None):

        callbacks = self._create_check_point()

        # compile first
        self.model.fit(x_train, validation_data = y_train, epochs=cfg.EPOCHS, 
                        steps_per_epoch=cfg.SPE, validation_steps=cfg.VS,
                        callbacks=callbacks) # to init 


    def load_wheights(self, path:str) -> None:

        self.model.load_weights(path)

    
    def model_save(self, path: str = "./models/super_resolution", weights_only: bool = True):

        if not weights_only:

            self.model.save(path)

        else:

            self.model.save_weights(path)


    def _create_check_point(self):

        filepath = './weights/sr_model_epoch{epoch:02d}.hdf5'
        checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=False,
                             mode='min') #save_freq=int(cfg.PERIOD * cfg.SPE)

        return [checkpoint]

        
    def _learning_rate(self):

        return PiecewiseConstantDecay(boundaries=cfg.BOUNDARIES, values=cfg.VALUES) # to init


    @staticmethod
    def _mixGE(y_true, y_pred):

        # y_hat, y_target = params

        lambda_param = cfg.LAMBDA_PARAM # to init

        #def filters
        sobel_x: list = [[-1, -2, -2], [0, 0, 0], [1, 2, 1]]
        sobel_y: list = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

        #cast
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        #create filters
        filter_x = tf.tile(
                tf.expand_dims(
                    tf.constant(sobel_x, dtype =tf.float32), axis = -1), 
                                                    [1, 1, K.shape(y_pred)[-1]])

        filter_x = tf.tile(tf.expand_dims(filter_x, axis = -1), [1, 1, 1, K.shape(y_pred)[-1]])


        filter_y = tf.tile(tf.expand_dims(
            tf.constant(sobel_y, dtype = tf.float32), axis = -1), 
                                                                        [1, 1, K.shape(y_true)[-1]])

        filter_y = tf.tile(tf.expand_dims(filter_y, axis = -1), [1, 1, 1, K.shape(y_true)[-1]])

        # output gradient
        output_gradient_x = tf.math.square(tf.nn.conv2d(y_pred, filter_x, strides = 1, padding = 'SAME'))
        output_gradient_y = tf.math.square(tf.nn.conv2d(y_pred, filter_y, strides = 1, padding = 'SAME'))

        #true gradient
        target_gradient_x = tf.math.square(tf.nn.conv2d(y_true, filter_x, strides = 1, padding = 'SAME'))
        target_gradient_y = tf.math.square(tf.nn.conv2d(y_true, filter_y, strides = 1, padding = 'SAME'))

        # square
        output_gradients = tf.math.sqrt(tf.math.add(output_gradient_x, output_gradient_y))
        target_gradients = tf.math.sqrt(tf.math.add(target_gradient_x, target_gradient_y))

        # compute mean gradient error
        shape = tf.cast(K.shape(output_gradients)[1:3], tf.float32)

        mge = tf.math.reduce_sum(tf.math.squared_difference(output_gradients, target_gradients) / (shape[0] * shape[1]))

        return tf.math.add(mean_squared_error(y_true, y_pred), tf.math.multiply(mge, lambda_param))


    def _summary(self):

        self.model.summary()


    def _eval(self, is_verbose: bool = True, x_test = None, y_test = None):

        preds = self.model.evaluate(x_test, y_test, batch_size=cfg.BATCH_SIZE, verbose=is_verbose, sample_weight=None)

        if is_verbose:
            
            self.model.summary()
            print()
            print ("Loss = " + str(preds[0]))
            print ("Test Accuracy = " + str(preds[1]))
            

    def _normalize(self, x):

        return tf.cast((x-self.mean_df), tf.float32)/self.std_df
        

    def _denormalize(self, x):

        return [x * self.std_df + value for value in self.mean_df]


    def _add_conv(self, x, name:str,\
                  filters: int=64,\
                  size: tuple = 3,\
                  strides: tuple = (1,1),\
                  padding: str = 'same'):

        return Conv2D(filters = filters, kernel_size = size, strides = strides, padding = padding, name=name)(x)


    def _add_resblock(self, x, block_num: int, batchQ: bool = cfg.BATCHNORM): # to init
        X_shortcut = x
        X = self._add_conv(x, strides = (1, 1), name=f'ConvLayer_{block_num}')

        if batchQ:

            X = BatchNormalization(name=f'BatchNorm_{block_num}')(X)

        X = self.activation_layer(name = f'ActivationLayer_{block_num}')(X)
        X = Add()([X, X_shortcut])

        return X


    def _non_linear_part(self, x):

        X = Lambda(self._normalize)(x)
        X = self._add_conv(X, name=f'Convolution_layer_0')
        through_shortcut = X

        for i in range(cfg.RES_NUM): # to init

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

        for i in range(cfg.CONV_NUM): # to init

            X = self._add_conv(X, name = f'Linear_Part_Conv_Layer_{i}', filters=3) # 3 for chanels

        X = Lambda(self._denormalize)(X)

        return X


    def _build_model(self):

        X_input = Input(cfg.INPUT_SHAPE) #to init
        X = self._non_linear_part(x = X_input)
        X = self._linear_part(X)

        # loss = Lambda(ModelBuilder._mixGE, name='mixGE')([X, X_input])

        model = Model(inputs=X_input, outputs=X, name='super_resolution')
        # model.add_loss(loss)

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
    DATA LOADER +
    ESRGAN
    """

        






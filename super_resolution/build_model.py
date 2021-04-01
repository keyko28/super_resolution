# res net16
from keras.layers import Conv2D, InputLayer, Add, Model, Lambda
from keras.layers import Conv2DTranspose, BatchNormalization, ReLU
from keras.models import load_model
from keras.optimizers import Adam
import config as cfg
from typing import Callable



class ModelBuilder:

    def __init__(self):
        pass


    def _normalize(self, x):

        return x/255.


    def _denormalize(self, x):

        return x * 255.


    def _add_conv(self, name:str,\
                  filters: int=32,\
                  size: tuple = (4, 4),\
                  strides: tuple = (2,2),\
                  padding: str = 'same'):

        return Conv2D(filters, size, strides, padding, name=name)


    def _add_resblock(self, block_num, batchQ: bool = True, activation_layer: Callable = ReLU):

        X_shortcut = X
        X = self._add_conv(name=f'ConvLayer {block_num}')(X)

        if batchQ:

            X = BatchNormalization(name=f'BatchNorm {block_num}')(X)

        X = activation_layer(name = f'ActivationLayer {block_num}')(X)
        X = Add()([X, X_shortcut])

        return X


    def _non_linear_part(self):

        res_num = 0 

        X = InputLayer((cfg.INPUT_SHAPE))
        X = self._add_conv(name=res_num)(X)

        for _ in range(cfg.RES_NUM):

            X = self._add_resblock(block_num = res_num)
            res_num += 1

        return X


    def _add_upscale(self, name:str,\
                  filters: int=1,\
                  size: tuple = (4, 4),\
                  strides: tuple = (2,2)):

        return Conv2DTranspose(filters, size, strides, name=name)


    def _linear_part(self, x):
        
        block_num = 0

        X = Conv2DTranspose(name = f'UPSACLE LAYER {block_num}')(X)

        for _ in range(cfg.CONV_NUM):

            X = self._add_conv(name = f'Linear Part Conv Layer {block_num}')(X)

        X = Lambda(self._denormalize)(X)

        return X


    def _build_model(self):

        X = self._normalize(X)

        X = self._non_linear_part()
        X = self._linear_part(X)

        model = Model(inputs=X_input, outputs=X, name='super_resolution')

        return model

    def fit_model(self, is_save:bool = False, is_verbose: bool = True):

        model = self.build_model()
        opt = Adam()

        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE)
        preds = model.evaluate(x_test, y_test, batch_size=cfg.BATCH_SIZE, verbose=is_verbose, sample_weight=None)

        if is_verbose:
            
            model.summary()
            print()
            print ("Loss = " + str(preds[0]))
            print ("Test Accuracy = " + str(preds[1]))
            
        if is_save:

            model.save('supply_chain_model_3')

        return preds

    def _load_model(self, model_name: str):
        """
        cd and model location must be equal
        """

        return load_model(model_name)

    def predict(self, data, model):

        model.predict(data)

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
    """

        






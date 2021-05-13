from tensorflow.python.keras.backend_config import epsilon
from generator import Generator
from discriminator import Discriminator
import config as cfg
from keras.layers import Input
from build_loss import Loss
from keras.models import Model
from keras.optimizers import Adam


class GanModel:

    def __init__(self) -> None:
        
        # discriminator and generator should be passed into the GanModel
        # as already inited and built models
        self.shape = cfg.INPUT_SHAPE
        self.model = None
        self.loss_weights = cfg.LOSS_WEIGHTS
        self.loss = None
        self.optimizer = None
        self.generator = None
        self.discriminator = None

        #opt params
        self.lr = cfg.LEARNING_RATE
        self.beta_1 = cfg.BETA_1
        self.beta_2 = cfg.BETA_2
        self.epsilon = cfg.EPSILON


    def _build_optimizer(self):
        return Adam(learning_rate=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)


    def _build_gan(self,
                generator: Generator,
                discriminator: Discriminator,
                loss = Loss):

        self.generator = generator
        self.discriminator = discriminator

        self.optimizer = self._build_optimizer()

        self.generator.model.compile(loss=loss._compile_loss, optimizer = self.optimizer)
        self.discriminator.model.compile(loss='binary_crossentropy', optimizer = self.optimizer)

        self.discriminator.model.trainable = False
        gan_input = Input(shape=self.shape)

        gen = self.generator.model(gan_input)
        gan_output = self.discriminator.model(gen)

        self.model = Model(inputs = gan_input, outputs=[gen, gan_output]) # gan model

        return self.model


    def _compile(self):

        self.model.compile(loss=[self.loss, 'binary_crossentropy'],
                           loss_weights=self.loss_weights,
                           optimizer=self.optimizer)


    def _summary(self):

        self.model.summary()




 
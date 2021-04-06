####################EXPERIMENTS#######################

CONV_NUM = 2
RES_NUM = 8
INPUT_SHAPE = (None, None, 3)
LEARNING_RATE = 0.00001
EPOCHS = 100
BATCH_SIZE = 32

MEAN = [value * 255 for value in [0.4488, 0.4371, 0.4040]]
STD = 127.5

SPE = 3000
VS = 10
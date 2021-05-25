import build_model as bm
import config as cfg
from data import load
from keras.layers import LeakyReLU, PReLU
from sr_esrgan.build_loss import Loss

def main() -> None:

    X_train, X_test = load()

    loss = Loss()

    model = bm.ModelBuilder(mean_df = cfg.MEAN, std_df = cfg.STD)

    # model.compile(loss=bm.ModelBuilder._mixGE)
    model.compile(loss=loss._compile_loss)
    model._summary()
    model.model_train(X_train, X_test)

    # save_results
    model.model_save(path = 'D:\pet_projects\super_resolution\weights\perceptual.h5', weights_only=True)

if __name__ == '__main__':
    main()




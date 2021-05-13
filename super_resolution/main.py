import build_model as bm
import config as cfg
from data import load
from keras.layers import LeakyReLU, PReLU

def main() -> None:

    X_train, X_test = load()

    model = bm.ModelBuilder(mean_df = cfg.MEAN, std_df = cfg.STD)

    model.compile(loss=bm.ModelBuilder._mixGE)
    model._summary()
    model.model_train(X_train, X_test)

    # save_results
    model.model_save(path = './models/mixGEPRELU/weights.h5', weights_only=True)

if __name__ == '__main__':
    main()




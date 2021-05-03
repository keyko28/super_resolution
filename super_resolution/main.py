import build_model as bm
import config as cfg
from data import load

X_train, X_test = load()

model = bm.ModelBuilder(cfg.MEAN, cfg.STD)

model.compile()
model._summary()
model.model_train(X_train, X_test)

# save_results
model.model_save()






import build_model as bm

model = bm.ModelBuilder(100, 2200)

model.compile()
model._summary()
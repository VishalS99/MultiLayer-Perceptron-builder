from mlp import MLP, Hidden, Loss, Output
from aux_utils import *

# Using 3 nodes in hidden layer

NN = MLP()
NN.add_layer('Hidden', dim_in=2, dim_out=3)
NN.add_layer('Hidden', dim_in=3, dim_out=3)
NN.add_layer('Hidden', dim_in=3, dim_out=3)
NN.add_layer('Output', dim_in=3, dim_out=3)
NN.add_layer('Loss', dim_in=3, dim_out=3)

loss, val_loss = NN.train(X_train, y_train, X_val, y_val, epochs=100, bsize=32, lr=0.05, alpha = 0.01)
plot_loss(loss, 'Loss', 100)
plot_loss(val_loss, 'Validation Loss', 100)
plot_decision_boundary(NN)

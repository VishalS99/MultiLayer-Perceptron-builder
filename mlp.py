import numpy as np

class Output:
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weights = np.random.uniform(size= (dim_out, dim_in))
        self.bias = np.random.uniform(size= (1, dim_out))

    def return_weights(self):
        return self.weights

    def forward_pass(self, prev_act):
        self.input = prev_act
        self.sum_wx = np.dot(prev_act, self.weights.T) + self.bias
        return self.sum_wx

    def backward_pass(self, prev_gradient, lr, alpha, N=1):
        self.gradients_W = np.dot( prev_gradient.T, self.input)
        self.gradient_b = prev_gradient

        # Performing weight updations, alpha is the regularization constant
        # Mini batch size has been divided to ensure overall gradient mean calculation 
        self.weights = self.weights*(1 - (alpha*(lr/N))) - (lr/N) * self.gradients_W
        self.bias = self.bias*(1 - (alpha*(lr/N))) - (lr/N) * self.gradient_b
        return prev_gradient
class Loss:
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out

    def softmax(self, x):
        e_x = np.exp(x)
        return e_x / np.sum(e_x)

    def forward_pass(self, y_hat, y, weight_norm, alpha):
        
        return -y_hat.flat[int(y.item(0))] + np.log(np.sum(np.exp(y_hat))) + (alpha/2) * (np.sum(weight_norm))
    
    def backward_pass(self, y_hat, y):
        o = self.softmax(y_hat)
        o.flat[int(y.item(0))] -= 1.0
        return o
    
class Hidden:
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weights = np.random.uniform(size= (dim_out, dim_in))
        self.bias = np.random.uniform(size= (1, dim_out))
        
    def return_weights(self):
        return self.weights
        
    def forward_pass(self, prev_act):
        self.input = prev_act
        self.sum_wx = np.dot(self.input, self.weights.T) + self.bias
        self.act = np.maximum(np.zeros(self.sum_wx.shape), self.sum_wx)
        return self.act

    def backward_pass(self, prev_gradient, prev_weights, lr, alpha, N=1):
        relu_derv = (self.sum_wx > 0) * 1
        temp = np.multiply(np.dot(prev_gradient, prev_weights) ,relu_derv)
        self.gradients_W = np.dot((temp).T, self.input)
        self.gradient_b = temp*1

        # Performing weight updations, alpha is the regularization constant
        # Mini batch size has been divided to ensure overall gradient mean calculation 
        self.weights = self.weights*(1 - (alpha*(lr/N))) - (lr/N) * self.gradients_W
        self.bias = self.bias*(1 - (alpha*(lr/N))) - (lr/N) * self.gradient_b
        return temp


class MLP:
    def __init__(self):
        self.hidden_layer = []
        self.output_layer = []
        self.loss_layer = []
    # Shuffler shuffles training and testing data
    
    def shuffler(self, X, y): 
        shuffle = np.random.permutation(len(X))
        return X[shuffle], y[shuffle]

    def add_layer(self, type, dim_in, dim_out):
        if type == 'Hidden':
            self.hidden_layer.append(Hidden(dim_in, dim_out))
        elif type == 'Output':
            
            self.output_layer = Output(dim_in, dim_out)
        elif type == 'Loss':
            self.loss_layer = Loss(dim_in, dim_out)
        else:
            return "Wrong Layer type"

    def train(self, X, y, X_val, y_val, epochs, bsize, lr, alpha):
        loss = []
        val_loss = []

        for e in range(epochs):
            err = 0
            X_shuffle, y_shuffle = self.shuffler(X, y)
            
            # Running iterations per batch
            for i in range(len(X_shuffle)//bsize):
                X_batch = X_shuffle[i*bsize:(i+1)*bsize]
                y_batch = y_shuffle[i*bsize:(i+1)*bsize]
                for j in range(len(X_batch)):
                    # forward pass
                    out = X_batch[j]
                    for k in self.hidden_layer:
                        out = k.forward_pass(out)
                    final = self.output_layer.forward_pass(out)
                    weight_norm = [np.linalg.norm(w.return_weights()) for w in self.hidden_layer]
                    weight_norm.append(np.linalg.norm(self.output_layer.return_weights()))
                    # Calculating err
                    err += self.loss_layer.forward_pass(final, y_batch[j], weight_norm, alpha)

                    # backpropagation:
                    grad = self.loss_layer.backward_pass(final, y_batch[j])
                    grad = self.output_layer.backward_pass(grad, lr, alpha,len(X_batch))
                    prev_weights = self.output_layer.return_weights()
                    for k in range(len(self.hidden_layer), 0, -1):
                        grad = self.hidden_layer[k-1].backward_pass(grad, prev_weights, lr, alpha, len(X_batch))
                        prev_weights = self.hidden_layer[k-1].return_weights()
   
            count = 0
            # Calculate loss on validation set
            val_pred = self.test(X_val, y_val)
            for i in range(len(y_val)):
                if val_pred[i] == y_val[i]: count+=1
            val_err = 1 - count/len(y_val)

            val_loss.append(val_err)
            loss.append(err / len(X))
            print("Epoch: {}/{}, Loss: {}, Val Loss: {}".format(e+1, epochs, loss[-1], val_loss[-1]))

            '''
             Early stopping - Check for 5 epochs after 10 epochs have passed
             If either the change in validation err is more than 0.005 or remains 0 (no val err change),
             we stop training
            '''
            count = 0
            if e > 10:
                for i in range(5):
                    if val_loss[-(i+1)] - val_loss[-(i+2)] >= 0.005:
                        count +=1
                    if val_loss[-(i+1)] - val_loss[-(i+2)] == 0:  count += 1
                    else: count = 0
                if count == 5: break
                else: continue
        
        return loss, val_loss

    def test(self, X, y):
        pred = []
        for i in range(len(X)):
            out = X[i]
            for j in self.hidden_layer:
                out = j.forward_pass(out)
            final = self.output_layer.forward_pass(out)
            pred_y = np.argmax(np.exp(final) / np.sum(np.exp(final)))
            pred.append(pred_y)
        return pred

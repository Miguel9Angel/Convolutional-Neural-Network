import numpy as np
import abc 

class CrossEntropyLoss():
        def forward(self, y_true, y_pred):
            epsilon = 1e-15  
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        def backward(self, y_true, y_pred):
            return y_pred-y_true
        
class Network():
    def __init__(self, layers):
        self.layers = layers
        self.loss_fn = CrossEntropyLoss()
    
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self, X, y, epochs, batch_size, learning_rate, validation_data=None):
        n = len(X)
        evaluation_cost, evaluation_accuracy = [], []
        for epoch in range(epochs):
            batches_X = [np.array(X[i:i+batch_size]) for i in range(0, n, batch_size)]
            batches_y = [np.array(y[i:i+batch_size]) for i in range(0, n, batch_size)]
            
            
            for batch in range(len(batches_X)):
                batch_X = batches_X[batch]
                batch_y = batches_y[batch]
                y_pred = self.forward(batch_X)

                loss = self.loss_fn.forward(batch_y, y_pred)

                grad = self.loss_fn.backward(batch_y, y_pred)
                self.backward(grad)
                
                for layer in self.layers:
                    if isinstance(layer, DenseLayer):
                        layer.weights -= learning_rate * layer.weights_gradient
                        layer.biases -= learning_rate * layer.biases_gradient
                    elif isinstance(layer, ConvLayer):
                        layer.kernels -= learning_rate * layer.dW
                        layer.bias    -= learning_rate * layer.db
            
            eval_cost, eval_accuracy = self.evaluate(validation_data)
            evaluation_accuracy.append(eval_accuracy)
            evaluation_cost.append(eval_cost)
            print(f"Epoch {epoch+1}, Loss: {loss}")
        
        return evaluation_cost, evaluation_accuracy
    
    def predict(self, X):
        return self.forward(X)
    
    def evaluate(self, test_data, threshold=0.5):
        X, y = test_data
        preds = self.predict(X)
        cost = self.loss_fn.forward(preds, y)
        
        if preds.shape[1] == 1:
            preds_bin = (preds >= threshold).astype(int)
            acc = np.mean(preds_bin == y)
        else:
            y_true = np.argmax(y, axis=1)
            y_pred = np.argmax(preds, axis=1)
            acc = np.mean(y_pred == y_true)
        return cost, acc 

class Layer(abc.ABC):
    def __init__(self):
        self.input = None
        self.output = None
    
    @abc.abstractmethod
    def forward(self, input):
        raise NotImplementedError
    
    @abc.abstractmethod
    def backward(self, output_gradient):
        raise NotImplementedError
    

class ConvLayer(Layer):
    def __init__(self, seed=42, kernel_size=2, filters=1, padding=0):
        super().__init__()
        np.random.seed(seed)        
        
        self.kernel_size = kernel_size
        self.filters = filters
        self.padding = padding
        self.kernels = None
        self.bias = None

    def relu(self, z):
        return np.maximun(0, z)
    
    def relu_derivate(self, z):
        return np.array(z>0, dtype=np.float64)
        
    def forward(self, X):
        self.X = X
        B, H, W, C = self.X.shape
        Kh, Kw, F =  self.kernel_size, self.kernel_size, self.filters
        
        if self.kernels is None:
            self.kernels = np.random.normal(0,1,size=(Kh,Kw,C,F))
            self.bias = np.zeros(F)
        
        X_padded = np.pad(
            X,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode="constant"
        )
        self.X_padded = X_padded

        H_out = H - Kh + 1 + 2*self.padding
        W_out = W - Kw + 1 + 2*self.padding
        out = np.zeros((B, H_out, W_out, F))
        
        for b in range(B):
            for f in range(self.filters):
                for i in range(0, H_out):
                    for j in range(0, W_out):
                        window = X_padded[b, i:i+Kh, j:j+Kw, :]
                        out[b,i,j,f] = np.sum(window*self.kernels[:,:,:,f]) + self.bias[f]
        
        self.out = out
        return out

    def backward(self, dY):
        B, H_out, W_out, F = dY.shape
        _, Xp_h, Xp_w, C = self.X.shape
        Kh, Kw, _, F = self.kernels.shape

        dX_padded = np.zeros_like(self.X_padded)
        self.dW = np.zeros_like(self.kernels)
        self.db = np.zeros_like(self.bias)

        for b in range(B):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        grad_val = dY[b,i,j,f]
                        window = self.X_padded[b, i:i+Kh, j:j+Kw, :]

                        self.dW[:,:,:,f] += window * grad_val
                        dX_padded[b, i:i+Kh, j:j+Kw, :] += self.kernels[:,:,:,f] * grad_val
                        self.db[f] += grad_val
        
        self.dW /= B
        self.db /= B
        if self.padding > 0:
            dX = dX_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dX = dX_padded
        
        return dX


class PoolingLayer(Layer):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size
        self.mask_gradient = None
        self.input_shape = None
        
    def forward(self, data):
        self.input_shape = data.shape
        B, H, W, C = self.input_shape
        pool_h, pool_w = self.pool_size, self.pool_size
        
        pool_data_h = H//pool_h
        pool_data_w = W//pool_w
        
        max_pooling = np.zeros((B, pool_data_h, pool_data_w, C))
        self.mask_gradient = np.zeros_like(max_pooling, dtype=int)
        
        for b in range(B):
            for f in range(C):
                for i in range(pool_data_h):
                    for j in range(pool_data_w):
                        window = data[b, i*pool_h:(i+1)*pool_h, j*pool_w:(j+1)*pool_w, f]
                        max_pooling[b, i, j, f] = np.max(window)
                        self.mask_gradient[b, i, j, f] = np.argmax(window)
        return max_pooling

    def backward(self, output_gradient):
        B, H_out, W_out, C = output_gradient.shape
        input_gradient = np.zeros(self.input_shape)

        for b in range(B):
            for f in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        grad = output_gradient[b, i, j, f]
                        max_index = self.mask_gradient[b, i, j, f]

                        row = (i * self.pool_size) + (max_index // self.pool_size)
                        col = (j * self.pool_size) + (max_index % self.pool_size)

                        input_gradient[b, row, col, f] = grad

        return input_gradient
    
class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.input = None
        
    def forward(self, data):
        self.input_shape = data.shape
        self.input = data
        return data.reshape(data.shape[0], -1)
                
    def backward(self, output_gradient):
        return output_gradient.reshape(self.input_shape)

class DenseLayer(Layer):
    def __init__(self, neurons, seed=42, activation='softmax'):
        super().__init__()
        np.random.seed(seed)
        self.weights = None
        self.biases = None
        self.neurons = neurons

        activations = {
            'relu': self.relu, 
            'sigmoid': self.sigmoid,
            'softmax': self.softmax
        }

        self.activation =  activations[activation]
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivate(self, z):
        derivate = np.array(z>0, dtype=np.float64)
        return derivate
    
    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def sigmoid_derivate(self, a):
        return a*(1-a)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, input_data):
        input_shape = input_data.shape
        if self.weights is None:
            self.weights = np.random.randn(input_shape[1], self.neurons)/np.sqrt(input_shape[1]) 
            self.biases = np.zeros((1, self.neurons))

        self.input = input_data
        z = np.dot(self.input, self.weights)+self.biases
        self.activation_output = self.activation(z)
        return self.activation_output
                
    def backward(self, output_gradient):
        B = self.input.shape[0]
        self.weights_gradient = np.dot(self.input.T, output_gradient)/B
        self.biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)/B

        input_gradient = np.dot(output_gradient, self.weights.T)

        return input_gradient

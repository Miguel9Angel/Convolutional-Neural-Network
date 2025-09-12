import numpy as np
import abc
class Network():
    def __init__(self, layers):
        self.layers = layers
        self.loss_fn = self.cross_entropy_cost
    
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)

            loss = self.loss_fn.forward(y, y_pred)

            grad = self.loss_fn.backward(y, y_pred)
            self.backward(grad)

            print(f"Epoch {epoch+1}, Loss: {loss}")
            
    def cross_entropy_cost(self, y_true, y_pred, derivate=False):
        def backward(y_true, y_pred):
            return y_pred-y_true

        def forward(y_true, y_pred):
            epsilon = 1e-15  
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
    def predict(self, X):
        return self.forward(X)

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
        self.kernels = np.random.normal(loc=0, scale=1, size=(kernel_size, kernel_size, filters))
        self.padding = padding
        self.bias = np.zeros(filters)

    def relu(self, z):
        return np.maximun(0, z)
    
    def relu_derivate(self, z):
        derivate = np.array(z>0, dtype=np.float64)
        return derivate
        
    def forward(self, X):
        self.X = X
        H, W, _ = self.X.shape
        Kh, Kw =  self.kernel_size, self.kernel_size
        
        X_padded = np.pad(X, self.padding, mode='constant')
        self.X_padded = X_padded
        H_out= H - Kh + 1
        W_out= W - Kw + 1
        out = np.zeros((H_out, W_out, self.filters))
        
        for f in range(self.filters):
            for i in range(0, H_out):
                for j in range(0, W_out):
                    window = X_padded[i:i+Kh, j:j+Kw, f]
                    out[i,j,f] = np.sum(window*self.kernels[:,:,f]) + self.bias[f]
        
        self.out = out
        return out

    def backward(self, dY):
        X_padded = self.X_padded
        Xp_h, Xp_w = X_padded.shape
        Kh, Kw, F = self.kernels.shape
        H_out, W_out, _ = dY.shape

        dX_padded = np.zeros_like(X_padded)
        dW = np.zeros_like(self.kernels)
        db = np.zeros_like(self.bias)

        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    grad_val = dY[i,j,f]
                    window = X_padded[i:i+Kh, j:j+Kw]

                    dW[:,:,f] += window*grad_val
                    dX_padded[i:i+Kh, j:j+Kw] += self.kernels[:,:,f]*grad_val
                    db[f] += grad_val
        
        if self.padding > 0:
            dX = dX_padded[self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
        
        return dX, dW, db


class PoolingLayer(Layer):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size
        self.mask_gradient = None
        self.input_shape = None
        
    def forward(self, data):
        self.input_shape = data.shape
        data_h, data_w, data_f = self.input_shape
        pool_data_h = data_h//self.pool_size
        pool_data_w = data_w//self.pool_size
        
        max_pooling = np.zeros((pool_data_h, pool_data_w, data_f))
        self.mask_gradient = np.zeros_like(max_pooling, dtype=int)
        
        for f in range(data_f):
            for i in range(0, data_h - self.pool_size+1, self.pool_size):
                for j in range(0, data_w - self.pool_size+1, self.pool_size):
                    a = i//self.pool_size
                    b = j//self.pool_size
                    window = data[i:i+self.pool_size, j:j+self.pool_size, f]
                    max_pooling[a, b, f] = np.max(window)
                    self.mask_gradient[a,b,f] = np.argmax(window)

        return max_pooling

    def backward(self, output_gradient):
        mask_flat = self.mask_gradient.flatten()
        output_gradient_flat = output_gradient.flatten()
        input_gradient_flat = np.zeros(self.input_shape).flatten()

        for i in range(len(output_gradient_flat)):
            gradient_value = output_gradient_flat[i]
            max_index_flat = mask_flat[i]
            input_gradient_flat[max_index_flat] = gradient_value

        input_gradient = input_gradient_flat.reshape(self.input_shape)

        data_h, data_w, data_f = self.input_shape
        input_gradient = np.zeros(self.input_shape)

        for f in range(data_f):
            for i in range(0, data_h-self.pool_size+1, self.pool_size):
                for j in range(o, data_w-self.pool_size+1, self.pool_size):
                    a = i // self.pool_size
                    b = j // self.pool_size

                    grad = output_gradient[a,b,f]
                    max_index = self.mask_gradient[a,b,f]

                    max_row = max_index // self.pool_size
                    max_col = max_index % self.pool_size
                    input_gradient[i+max_row, j+max_col] = grad

        return input_gradient
    
class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.input = None
        
    def forward(self, data):
        self.input_shape = data.shape
        self.input = data
        print(self.input_shape)
        return self.input.flatten()
                
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
        return np.maximun(0, z)
    
    def relu_derivate(self, z):
        derivate = np.array(z>0, dtype=np.float64)
        return derivate
    
    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def sigmoid_derivate(self, a):
        return a*(1-a)
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, input_data):
        input_shape = input_data.shape
        print(input_shape)
        self.weights = np.random.randn(input_shape[1], self.neurons)/np.sqrt(input_shape) 
        self.biases = np.zeros((1, self.neurons))

        self.input = input_data
        z = np.dot(self.input, self.weights)+self.biases
        self.activation_output = self.activation(z)
        return self.activation_output
                
    def backward(self, output_gradient):
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        input_gradient = np.dot(output_gradient, self.weights.T)

        return input_gradient, weights_gradient, biases_gradient

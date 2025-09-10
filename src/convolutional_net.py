import numpy as np
import abc
class Network():
    def __init__(self):
        self.layers = []
        self.cost_function = 0
        self.optimizer = 0
    
    def train():
        pass
    
    def predict():
        pass

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
    def __init__(self, seed=42, kernel_size=1, filters=1, padding=0):
        super().__init__()
        np.random.seed(seed)        
        
        self.kernel_size = kernel_size
        self.filters = filters
        self.kernels = np.random.normal(loc=0, scale=1, size=(self.kernel_size, self.kernel_size, self.filters))
        self.padding = padding
        self.bias = np.zeros(filters)

    def relu(self, z):
        return np.maximun(0, z)
    
    def relu_derivate(self, z):
        derivate = np.array(z>0, dtype=np.float64)
        return derivate
        
    def forward(self, X):
        self.X = X
        H, W = self.X.shape
        X_padded = np.zeros((H+self.padding*2,W+self.padding*2))
        padded_rows_img, padded_cols_img = X_padded.shape
        X_padded[self.padding:padded_rows_img-self.padding, self.padding:padded_cols_img-self.padding] = self.data
        
        data_filtered = []
        for idx, kernel in enumerate(self.kernels):
            kernel_rows, kernel_cols = kernel.shape
            img_filter_rows = padded_rows_img-kernel_rows+1
            img_filter_cols = padded_cols_img-kernel_cols+1

            image_filter = np.zeros((img_filter_rows, img_filter_cols))
            
            for i in range(img_filter_rows):
                for j in range(img_filter_cols):
                    image_filter[i, j] = np.sum(padded_img[i:kernel_rows+i, j:kernel_cols+j]*kernel)+self.bias[idx]
            
            activated_filter = self.relu(image_filter)
            data_filtered.append(activated_filter)
        
        self.output = np.array(data_filtered)
        self.output = np.transpose(self.output, (1, 2, 0))
        
        return self.output
        
    def backward(self):
        pass

class PoolingLayer(Layer):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size
        self.mask_gradient = None
        self.input_shape = None
        
    def forward(self, data):
        self.input_shape = data.shape
        data_rows, data_cols = self.input_shape
        pool_data_rows = data_rows//self.pool_size
        pool_data_cols = data_cols//self.pool_size
        
        max_pooling = np.zeros((pool_data_rows, pool_data_cols))
        self.mask_gradient = np.zeros_like(max_pooling, dtype=int)

        for i in range(0, data_rows - self.pool_size+1, self.pool_size):
            for j in range(0, data_cols - self.pool_size+1, self.pool_size):
                a = i//self.pool_size
                b = j//self.pool_size
                window = data[i:i+self.pool_size, j:j+self.pool_size]
                max_pooling[a, b] = np.max(window)
                self.mask_gradient[a,b] = np.argmax(window)

        return max_pooling
                   
    def backward(self):
        pass
    
    def backward(self, output_gradient):
        input_gradient = np.zeros(self.input_shape)
        
        mask_flat = self.mask_gradient.flatten()
        output_gradient_flat = output_gradient.flatten()
        
        for i in range(len(output_gradient_flat)):
            gradient_value = output_gradient_flat[i]
            max_index_flat = mask_flat[i]
            
            row_index = i // (self.input_shape[1] // self.pool_size)
            col_index = i % (self.input_shape[1] // self.pool_size)
            
            max_row_in_window = max_index_flat // self.pool_size
            max_col_in_window = max_index_flat % self.pool_size
            
            original_row = row_index * self.pool_size + max_row_in_window
            original_col = col_index * self.pool_size + max_col_in_window
            input_gradient[original_row, original_col] = gradient_value
        return input_gradient
    
class FlattenLayer(Layer):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.input = None
        
    def forward(self, data):
        self.input = data
        return self.input.flatten()
                
    def backward(self, output_gradient):
        return output_gradient.reshape(self.input_shape)

class DenseLayer(Layer):
    def __init__(self, neuronas, seed=42, activation='softmax', input_shape=None):
        super().__init__()
        np.random.seed(seed)
        self.weights = np.random.randn(input_shape, neuronas)/np.sqrt(input_shape) 
        self.biases = np.zeros((1, neuronas))

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
        self.input = input_data
        z = np.dot(input, self.weights)+self.biases
        self.activation_output = self.activation(z)
        return self.activation_output
                
    def backward(self, output_gradient):
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        input_gradient = np.dot(output_gradient, self.weights.T)

        return input_gradient, weights_gradient, biases_gradient

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
    def __init__(self, seed=42, data, kernel_size=1, filters=1, padding=0, stride=1):
        super().__init__()
        np.random.seed(seed)        
        self.data = data
        self.kernel_size = kernel_size
        self.filters = filters
        self.padding = padding
        self.stride = stride
        self.bias = np.zeros(filters)
        self.output = None

    def relu(self, z):
        return np.maximun(0, z)
    
    def relu_derivate(self, z):
        derivate = np.array(z>0, dtype=np.float64)
        return derivate
        
    def forward(self):
        rows_img, col_img = self.data.shape
        padded_img = np.zeros((rows_img+self.padding*2,col_img+self.padding*2))
        padded_rows_img, padded_cols_img = padded_img.shape
        padded_img[self.padding:padded_rows_img-self.padding, self.padding:padded_cols_img-self.padding] = self.data
        
        
        kernels = np.random.normal(loc=0, scale=1, size=(self.kernel_size, self.kernel_size, self.filters))
        data_filtered = []
        for idx, kernel in enumerate(kernels):
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
        
    def forward(self):
        pass
                
    def backward(self):
        pass

class FlattenLayer(Layer):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        
    def forward(self):
        pass
                
    def backward(self):
        pass

class DenseLayer(Layer):
    def __init__(self):
        super().__init__()
        self.weights = None
        self.bias = None
        
    def forward(self):
        pass
                
    def backward(self):
        pass
import numpy as np

def conv2d(data, kernel_size=1, filters=1, padding=0, stride=1):
    rows_img, col_img = data.shape
    padded_img = np.zeros((rows_img+padding*2,col_img+padding*2))
    padded_rows_img, padded_cols_img = padded_img.shape
    padded_img[padding:padded_rows_img-padding, padding:padded_cols_img-padding] = data
    
    
    kernels = np.random.normal(loc=0, scale=1, size=(kernel_size, kernel_size, filters))
    bias = np.zeros
    data_filtered = []
    for kernel in kernels:
        kernel_rows, kernel_cols = kernel.shape
        img_filter_rows = padded_rows_img-kernel_rows+1
        img_filter_cols = padded_cols_img-kernel_cols+1

        image_filter = np.zeros((img_filter_rows, img_filter_cols))
        
        for i in range(img_filter_rows):
            for j in range(img_filter_cols):
                image_filter[i, j] = np.sum(padded_img[i:kernel_rows+i, j:kernel_cols+j]*kernel)


    return data_filtered
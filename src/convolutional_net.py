import numpy as np

def conv2d(image, kernel, padding=0, stride=1):
    image = X_train[0]
    padding=1
    col_img, rows_img = image.shape
    padded_img = np.zeros((rows_img+padding*2,col_img+padding*2))
    padded_img[padding:image.shape[0]+padding, padding:image.shape[1]+padding] = image
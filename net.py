import tensorflow as tf
import numpy as np
import scipy.ndimage.filters as fi


class Net:
    def __init__(self):
        pass
    
    def getGaussianPyramid(self, N, size=5, sigma=3.0):
        gaussian_kernel = tf.convert_to_tensor(generate_gaussian_kernel2d(self, size, sigma)) 
        def gaussianPyrDown(x):
            stacked = [x]
            for _ in range(N - 1):
                x = tf.nn.conv2d(x, gaussian_kernel, strides=[1,1,1,1], padding='VALID')
                stacked.append(x)
            return stacked
        return gaussianPyrDown

     def generate_gaussian_kernel2d(self, size = 5, sigma = 3.0):
        inp = np.zeros((size, size))
        inp[size//2, size//2] = 1
        return fi.gaussian_filter(inp, sigma)
       

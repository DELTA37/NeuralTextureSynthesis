import tensorflow as tf
import numpy as np
import scipy.ndimage.filters as fi


class Net:
    def __init__(self):
        self.inputs = tf.placeholder(dtype=tf.float32, name="inputs")
        self.gpyr = self.getGaussianPyramid(5)
        self.outputs = self.gpyr(self.inputs)
    
    def getGaussianPyramid(self, N, size=5, sigma=3.0):
        kernel = self.generate_gaussian_kernel2d(size, sigma) 
        print(kernel)
        gaussian_kernel = tf.convert_to_tensor(np.repeat(kernel, 3, axis=1).reshape((size, size, 3, 1)))
        def gaussianPyrDown(x):
            stacked = [x]
            for _ in range(N - 1):
                x = tf.nn.depthwise_conv2d(x, gaussian_kernel, strides=[1,1,1,1], padding='VALID')[:,::2,::2,:]
                stacked.append(x)
            return stacked
        return gaussianPyrDown

    def generate_gaussian_kernel2d(self, size = 5, sigma = 3.0):
        inp = np.zeros((size, size))
        inp[size//2, size//2] = 1
        return np.float32(fi.gaussian_filter(inp, sigma))
    
    def predict(self, images):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            results = sess.run(self.outputs, feed_dict = {self.inputs: images})
        return results





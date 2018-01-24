import tensorflow as tf
import numpy as np
import scipy.ndimage.filters as fi

from vgg19 import Vgg19

class Net:
    def __init__(self, vgg19_npy_path):
        self.vgg19 = Vgg19(vgg19_npy_path)
        self.gpyr = self.getGaussianPyramid(5)

        self.source_inputs = tf.placeholder(dtype=tf.float32)
        self.source_outputs = self.gpyr(self.source_inputs)
        self.source_encoded = [self.vgg19.build(output) for output in self.source_outputs]

    
    def getGaussianPyramid(self, N, size=5, sigma=3.0):
        kernel = self.generate_gaussian_kernel2d(size, sigma) 
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
    
    def getLoss(self, source_encoded, target_encoded):
        pass

    def predict(self, images):
        self.target_inputs = tf.Variable(np.float32(np.random.randn(*images.shape)))
        self.target_outputs = self.gpyr(self.target_inputs)
        self.target_encoded = [self.vgg19.build(output) for output in self.target_outputs]

        print(self.target_encoded[0][0])
        exit()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            results = sess.run(self.outputs, feed_dict = {self.inputs: images})
        return results





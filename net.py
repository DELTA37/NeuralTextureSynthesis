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

        self.source_encoded = []
        for output in self.source_outputs:
            self.source_encoded.extend(self.vgg19.build(output))

    
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
        loss = tf.Variable(0, dtype=tf.float32)
        for i in range(len(source_encoded)):
            x = source_encoded[i]
            x = tf.reshape(x, [-1, int(x.shape[3])])
            x = tf.matmul(x, x, transpose_a=True)

            y = target_encoded[i]
            y = tf.reshape(target_encoded[i], [-1, int(y.shape[3])])
            y = tf.matmul(y, y, transpose_a=True)
            loss += tf.reduce_sum((x - y)**2)

        return loss

    def predict(self, images, N=1000):
        self.target_inputs = tf.Variable(np.float32(np.random.randn(*images.shape)))
        self.target_outputs = self.gpyr(self.target_inputs)

        self.target_encoded = []
        for output in self.target_outputs:
            self.target_encoded.extend(self.vgg19.build(output))
        
        loss = self.getLoss(self.source_encoded, self.target_encoded)

        init = tf.global_variables_initializer()
        opt = tf.train.AdamOptimizer(0.01).minimize(loss, var_list=[self.target_inputs])

        with tf.Session() as sess:
            sess.run(init)
            for i in range(N):
                _loss, _, results = sess.run([loss, opt, self.target_inputs], feed_dict = {self.source_inputs: images})
                print(_loss)
        return results


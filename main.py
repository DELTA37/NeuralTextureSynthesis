import cv2
import argparse
from net import Net
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)

    args = parser.parse_args()
    path = args.path
    images = np.float32(cv2.imread(path, cv2.IMREAD_COLOR)[np.newaxis, :, :, :] / 255.0) 
    print(images.shape)

    net = Net('./vgg19.npy')
    res = net.predict(images)

    print(res[0].shape)
    print(res[1].shape)
    print(res[2].shape)
    print(res[3].shape)
    print(res[4].shape)
    plt.imshow(res[0][0])
    plt.show()
    plt.imshow(res[1][0])
    plt.show()
    plt.imshow(res[2][0])
    plt.show()
    plt.imshow(res[3][0])
    plt.show()
    plt.imshow(res[4][0])
    plt.show()

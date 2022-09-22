import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import sys
import random
from numba import jit
import math
import time

pathsep = "/"

def AddNoise(img):
    m, n = img.shape
    mean, var = 50., 10.0
    gaus = np.random.normal(mean, var, m * n).reshape(m, n)
    ret = img.copy()
    for i in range(m):
        for j in range(n):
            if ret[i,j] + gaus[i,j] < 255:
                ret[i,j] += gaus[i,j]
            else:
                ret[i,j] = 255
    return ret

def GenerateDither(dsize):
    if dsize == 2:
        return np.array([[1, 2], [3, 0]])
    else:
        submatrix = GenerateDither(round(dsize/2))
        lefthalf, righthalf = np.concatenate((4 * submatrix + 1, 4 * submatrix + 3), axis = 0), np.concatenate((4 * submatrix + 2, 4 * submatrix), axis = 0)
        return np.concatenate((lefthalf, righthalf), axis = 1)

def Dithering(img, dsize):
    noisy = AddNoise(img)
    # plt.imshow(noisy, cmap="gray")
    # plt.show()
    dithermatrix = GenerateDither(dsize)
    thresholdmaxtrix = 255 * (dithermatrix + 0.5) / dsize ** 2
    ret = np.zeros(img.shape, dtype=np.uint8)
    m, n = img.shape
    for i in range(0, m, dsize):
        for j in range(0, n, dsize):
            for k in range(dsize):
                for l in range(dsize):
                    if noisy[i+k,j+l] > thresholdmaxtrix[k,l]:
                        ret[i+k,j+l] = 255
                    else:
                        ret[i+k,j+l] = 0
    return ret

def Prob1():
    sample1 = cv.imread(pathsep.join([".", "hw4_sample_images", "sample1.png"]), cv.IMREAD_GRAYSCALE)
    if sample1 is None:
        sys.exit("Can't open {}".format(pathsep.join(["hw4_sample_images", "sample1.png"])))
    res1 = Dithering(sample1, 2)
    cv.imwrite(pathsep.join([".", "result1.png"]), res1)
    res2 = Dithering(sample1, 256)
    cv.imwrite(pathsep.join([".", "result2.png"]), res2)

Prob1()
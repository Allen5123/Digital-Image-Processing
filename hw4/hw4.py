import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import sys
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

@jit
def MirrorPadding(img, ksize):
    m, n = img.shape
    s, t = ksize
    #mirror padding
    padded = np.zeros((m+(s//2)*2, n+(t//2)*2), dtype=np.int32)
    padded[s//2:s//2+m, t//2:t//2+n] = img[:,:]
    padded[:s//2,t//2:t//2+n] = padded[(s//2)*2:s//2:-1,t//2:t//2+n]
    padded[-1:-(s//2+1):-1,t//2:t//2+n] = padded[-((s//2)*2+1):-(s//2+1),t//2:t//2+n]
    padded[:,:t//2] = padded[:,2*(t//2):t//2:-1]
    padded[:,-1:-(t//2+1):-1] = padded[:,-(2*(t//2)+1):-(t//2+1)]
    return padded

@jit
def ErrorDiffusion(img, pattern):
    errormatrix = img.astype(np.float32)
    m, n = img.shape
    ltor, rtol = pattern, np.fliplr(pattern)
    for i in range(m):
        if i % 2 == 0:
            for j in range(n):
                newpixel = 255 * np.round(errormatrix[i,j] / 255)
                error = errormatrix[i,j] - newpixel
                errormatrix[i,j] = newpixel
                s, t = ltor.shape
                for k in range(-(s//2), s//2+1):
                    for l in range(-(t//2), t//2+1):
                        if i + k >= 0 and i + k < m and j + l >= 0 and j + l < n:
                            errormatrix[i+k,j+l] += error * ltor[k+s//2,l+t//2]
        else:
            for j in range(n-1,-1,-1):
                newpixel = 255 * np.round(errormatrix[i,j] / 255)
                error = errormatrix[i,j] - newpixel
                errormatrix[i,j] = newpixel
                s, t = rtol.shape
                for k in range(-(s//2), s//2+1):
                    for l in range(-(t//2), t//2+1):
                        if i + k >= 0 and i + k < m and j + l >= 0 and j + l < n:
                            errormatrix[i+k,j+l] += error * rtol[k+s//2,l+t//2]
    return errormatrix.astype(np.uint8)

def Prob1():
    sample1 = cv.imread(pathsep.join([".", "hw4_sample_images", "sample1.png"]), cv.IMREAD_GRAYSCALE)
    if sample1 is None:
        sys.exit("Can't open {}".format(pathsep.join(["hw4_sample_images", "sample1.png"])))
    res1 = Dithering(sample1, 2)
    cv.imwrite(pathsep.join([".", "result1.png"]), res1)
    res2 = Dithering(sample1, 256)
    cv.imwrite(pathsep.join([".", "result2.png"]), res2)
    res3 = ErrorDiffusion(sample1, np.array([[0.,0.,0.],[0.,0.,7.],[3.,5.,1.]])/16.)
    cv.imwrite(pathsep.join([".", "result3.png"]), res3)
    res4 = ErrorDiffusion(sample1, np.array([[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,7.,5.],[3.,5.,7.,5.,3.],[1.,3.,5.,3.,1.]])/48.)
    cv.imwrite(pathsep.join([".", "result4.png"]), res4)
    atkinson = ErrorDiffusion(sample1, np.array([[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,1.,1.],[0.,1.,1.,1.,0.],[0.,0.,1.,0.,0.]])/8.)
    cv.imwrite(pathsep.join([".","atkinson.png"]), atkinson)

def Sampling(img, tx, ty):
    m, n = img.shape
    sampleimg = []
    for i in range(0, m, tx):
        sampleimg.append([])
        for j in range(0, n, ty):
            sampleimg[-1].append(img[i,j])
    imghat = np.fft.fft2(sampleimg)
    # plt.imshow(np.log10(np.abs(imghat))), plt.colorbar()
    # plt.show()
    return np.fft.ifft2(imghat).real

def Padding(img, kernel):
    m, n = img.shape
    s, t = kernel.shape
    #mirror padding
    padded = np.zeros((m+(s//2)*2, n+(t//2)*2), dtype=img.dtype)
    padded[s//2:s//2+m, t//2:t//2+n] = img[:,:]
    padded[:s//2,t//2:t//2+n] = padded[(s//2)*2:s//2:-1,t//2:t//2+n]
    padded[-1:-(s//2+1):-1,t//2:t//2+n] = padded[-((s//2)*2+1):-(s//2+1),t//2:t//2+n]
    padded[:,:t//2] = padded[:,2*(t//2):t//2:-1]
    padded[:,-1:-(t//2+1):-1] = padded[:,-(2*(t//2)+1):-(t//2+1)]
    return padded

def Conv(img, kernel):
    m, n = img.shape
    s, t = kernel.shape
    ret = np.zeros_like(img)
    padded = Padding(img, kernel)
    for i in range(m):
        for j in range(n):
            for k in range(-(s//2), s//2+1):
                for l in range(-(t//2), t//2+1):
                    ret[i,j] += padded[s//2+i+k,t//2+j+l] * kernel[s//2+k,t//2+l]
    return ret

def Gaussian(sz):
    gauss = np.empty((sz, sz))
    sigma = sz / 6
    summ = 0
    for i in range(sz):
        for j in range(sz):
            gauss[i,j] = math.exp(-((i-sz//2)**2+(j-sz//2)**2) / (2*(sigma**2)))        
            summ += gauss[i,j]
    gauss /= summ
    return gauss

def UnsharpInFreq(img, gsz, c):
    imghat, gaussianhat = np.fft.fft2(img), np.fft.fft2(Gaussian(gsz))
    filted = imghat * gaussianhat
    imghatunsharp = c / (2 * c - 1) * imghat - (1 - c) / (2 * c - 1) * filted
    plt.subplot(121)
    plt.imshow(np.log10(np.abs(imghat)), cmap="gray"), plt.axis("off")
    plt.subplot(122)
    plt.imshow(np.log10(np.abs(imghatunsharp)), cmap="gray"), plt.axis("off")
    plt.show()
    return np.fft.ifft2(imghatunsharp).real

def Prob2():
    sample2 = cv.imread(pathsep.join([".", "hw4_sample_images", "sample2.png"]), cv.IMREAD_GRAYSCALE)
    if sample2 is None:
        sys.exit("Can't open {}".format(pathsep.join(["hw4_sample_images", "sample2.png"])))
    sample3 = cv.imread(pathsep.join([".", "hw4_sample_images", "sample3.png"]), cv.IMREAD_GRAYSCALE)
    if sample3 is None:
        sys.exit("Can't open {}".format(pathsep.join(["hw4_sample_images", "sample3.png"])))
    result5 = Sampling(sample2, 3, 3)
    cv.imwrite(pathsep.join([".", "result5.png"]), result5)
    result6 = UnsharpInFreq(sample3, sample3.shape[0], 18./30.)
    cv.imwrite(pathsep.join([".", "result6.png"]), result6)

Prob1()
Prob2()
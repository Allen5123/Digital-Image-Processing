import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import sys
import time

pathsep = "/"

def SwapRB(imgs):
    for img in imgs:
        m, n, c = img.shape
        for i in range(m):
            for j in range(n):
                img[i,j,0], img[i,j,2] = img[i,j,2], img[i,j,0]

def Flip(img):
    flipped = np.empty_like(img)
    flipped[:][:] = img[::-1,:]
    return flipped

def ToGray(img):
    m, n = img.shape[0], img.shape[1]
    gray = np.empty((m, n), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            gray[i,j] = img[i,j,0] // 3 + img[i,j,1] // 3 + img[i,j,2] // 3
    return gray

def WarnUp():
    imgpath = pathsep.join([".", "hw1_sample_images", "sample1.png"])
    img = cv.imread(imgpath)
    if img is None:
        sys.exit("Can't open {}".format(imgpath))
        
    flipped = Flip(img)
    gray = ToGray(flipped)
    cv.imwrite(pathsep.join([".", "result1.png"]), flipped)
    cv.imwrite(pathsep.join([".", "result2.png"]), gray)
    
    # SwapRB([img, flipped])
    # plt.subplot(131), plt.imshow(img), plt.title("sample1")
    # plt.subplot(132), plt.imshow(flipped), plt.title("result1")
    # plt.subplot(133), plt.imshow(gray, cmap="gray"), plt.title("result2")
    # plt.show()

def MulIntensity(img, mul):
    ret = np.empty_like(img)
    m, n = img.shape
    for i in range(m):
        for j in range(n):
            if img[i,j] * mul >= 255:
                ret[i,j] = 255
            else:
                ret[i,j] = img[i,j] * mul
    return ret

def Histogram(img):
    hist = np.zeros(256)
    m, n = img.shape
    for i in range(m):
        for j in range(n):
            hist[img[i,j]] += 1
    #normalize
    hist /= (m * n)
    return hist

def GlobalHistEq(img):
    L = 256
    prefix = np.zeros(L)
    hist = Histogram(img)
    prefix[0] = hist[0]
    for i in range(1, L):
        prefix[i] = prefix[i-1] + hist[i]
    transform = prefix * (L - 1)
    ret = np.empty_like(img)
    m, n = img.shape
    for i in range(m):
        for j in range(n):
            ret[i,j] = transform[img[i,j]]
    return ret

#optimize
def LocalHistEq(img, ksize):
    m, n = img.shape
    s, t = ksize
    padded = Padding(img, np.empty(ksize))
    ret = np.empty(img.shape)
    prefix = np.zeros((padded.shape[0], padded.shape[1], 256), dtype=np.int64)
    for i in range(padded.shape[0]):
        for j in range(padded.shape[1]):
            prefix[i,j,padded[i,j]] += 1
            if i - 1 >= 0:
                prefix[i,j,:] += prefix[i-1,j,:]
            if j - 1 >= 0:
                prefix[i,j,:] += prefix[i,j-1,:]
            if i - 1 >= 0 and j - 1 >= 0:
                prefix[i,j,:] -= prefix[i-1,j-1,:] 
    for i in range(m):
        for j in range(n):
            rmin, rmax, cmin, cmax = (i, 2*(s//2)+i, j, 2*(t//2)+j)
            subarr = np.copy(prefix[rmax,cmax,:])
            if cmin - 1 >= 0:
                subarr -= prefix[rmax,cmin-1,:]
            if rmin - 1 >= 0:
                subarr -= prefix[rmin-1,cmax,:]
            if rmin - 1 >= 0 and cmin - 1 >= 0:
                subarr += prefix[rmin-1,cmin-1,:]
            cnt = 0
            for k in range(len(subarr)):
                if np.uint8(k) <= padded[s//2+i,t//2+j]:
                    cnt += subarr[k]
            ret[i,j] = 255 * cnt // (s * t)
    return ret.astype(np.uint8)

def Mytransform(img):
    localimg = LocalHistEq(img, (201, 151))
    c = 1
    ret = np.power(localimg / 255, 0.6) * 255 * c
    return ret.astype(np.uint8)
    
def Prob1():
    imgpath = pathsep.join([".", "hw1_sample_images", "sample2.png"])
    img = cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit("Can't open {}".format(imgpath))
        
    div2 = MulIntensity(img, 1 / 2)
    mul3 = MulIntensity(img, 3)
    cv.imwrite(pathsep.join([".", "result3.png"]), div2)
    cv.imwrite(pathsep.join([".", "result4.png"]), mul3)
    
    # imghist = Histogram(img)
    # div2hist = Histogram(div2)
    # mul3hist = Histogram(mul3)
    div2global = GlobalHistEq(div2)
    # div2glohist = Histogram(div2global)
    div2local = LocalHistEq(div2, (201, 151))
    # div2lochist = Histogram(div2local)
    mul3global = GlobalHistEq(mul3)
    # mul3glohist = Histogram(mul3global)
    mul3local = LocalHistEq(mul3, (201, 151))
    # mul3lochist = Histogram(mul3local)
    result9 = Mytransform(img)
    # result9hist = Histogram(result9)
    
    cv.imwrite(pathsep.join([".", "result5.png"]), div2global)
    cv.imwrite(pathsep.join([".", "result6.png"]), mul3global)
    cv.imwrite(pathsep.join([".", "result7.png"]), div2local)
    cv.imwrite(pathsep.join([".", "result8.png"]), mul3local)
    cv.imwrite(pathsep.join([".", "result9.png"]), result9)
    
    # plt.figure(1)
    # plt.subplot(111), plt.title("Histogram of sample2")
    # mk, st, bs = plt.stem(np.arange(256), imghist, basefmt=" ")
    # plt.setp(st, linewidth=0.4)
    # plt.setp(mk, markersize=0.7)
    # plt.figure(2)
    # plt.subplot(111), plt.title("Histogram of result3")
    # mk, st, bs = plt.stem(np.arange(256), div2hist, basefmt=" ")
    # plt.setp(st, linewidth=0.4)
    # plt.setp(mk, markersize=0.7)
    # plt.figure(3)
    # plt.subplot(111), plt.title("Histogram of result5")
    # mk, st, bs = plt.stem(np.arange(256), div2glohist, basefmt=" ")
    # plt.setp(st, linewidth=0.4, color="r")
    # plt.setp(mk, markersize=0.7, color="r")
    # plt.figure(4)
    # plt.subplot(111), plt.title("Histogram of result7")
    # mk, st, bs = plt.stem(np.arange(256), div2lochist, basefmt=" ")
    # plt.setp(st, linewidth=0.4, color="g")
    # plt.setp(mk, markersize=0.7, color="g")
    # plt.figure(5)
    # plt.subplot(111), plt.title("Histogram of result4")
    # mk, st, bs = plt.stem(np.arange(256), mul3hist, basefmt=" ")
    # plt.setp(st, linewidth=0.4)
    # plt.setp(mk, markersize=0.7)
    # plt.figure(6)
    # plt.subplot(111), plt.title("Histogram of result6")
    # mk, st, bs = plt.stem(np.arange(256), mul3glohist, basefmt=" ")
    # plt.setp(st, linewidth=0.4, color="r")
    # plt.setp(mk, markersize=0.7, color="r")
    # plt.figure(7)
    # plt.subplot(111), plt.title("Histogram of result8")
    # mk, st, bs = plt.stem(np.arange(256), mul3lochist, basefmt=" ")
    # plt.setp(st, linewidth=0.4, color="g")
    # plt.setp(mk, markersize=0.7, color="g")
    # plt.figure(8)
    # plt.subplot(111), plt.title("Histogram of result9")
    # mk, st, bs = plt.stem(np.arange(256), result9hist, basefmt=" ")
    # plt.setp(st, linewidth=0.4, color="g")
    # plt.setp(mk, markersize=0.7, color="g")
    # plt.show()
    
def Padding(img, kernel):
    m, n = img.shape
    s, t = kernel.shape
    #mirror padding
    padded = np.zeros((m+(s//2)*2, n+(t//2)*2), dtype=np.uint8)
    padded[s//2:s//2+m, t//2:t//2+n] = img[:,:]
    padded[:s//2,t//2:t//2+n] = padded[(s//2)*2:s//2:-1,t//2:t//2+n]
    padded[-1:-(s//2+1):-1,t//2:t//2+n] = padded[-((s//2)*2+1):-(s//2+1),t//2:t//2+n]
    padded[:,:t//2] = padded[:,2*(t//2):t//2:-1]
    padded[:,-1:-(t//2+1):-1] = padded[:,-(2*(t//2)+1):-(t//2+1)]
    return padded

def Conv(img, kernel):
    m, n = img.shape
    s, t = kernel.shape
    ret = np.zeros(img.shape, dtype=np.int32)
    padded = Padding(img, kernel)
    # print(padded)
    for i in range(m):
        for j in range(n):
            for k in range(-(s//2), s//2+1):
                for l in range(-(t//2), t//2+1):
                    if ret[i,j] + padded[s//2+i+k,t//2+j+l] * kernel[s//2+k,t//2+l] < 255:
                        ret[i,j] += padded[s//2+i+k,t//2+j+l] * kernel[s//2+k,t//2+l]
                    else:
                        ret[i,j] = 255
                        break;
                if ret[i,j] == 255:
                    break;
    return ret.astype(np.uint8)

def MedianFilter(img, kernelsz):
    m, n = img.shape
    s, t = kernelsz
    padded = Padding(img, np.empty((s, t)))
    ret = np.empty_like(img)
    for i in range(m):
        for j in range(n):
            subarr = []
            for k in range(-(s//2), s//2+1):
                for l in range(-(t//2), t//2+1):
                    subarr.append(padded[s//2+i+k, t//2+j+l])
            subarr.sort()
            ret[i,j] = subarr[len(subarr)//2]
    return ret

def Psnr(ori, res):
    m, n = ori.shape
    mse = np.sum(((res - ori) ** 2)) / (m * n)
    return 10 * np.log10((255. ** 2) / mse)

def Prob2():
    imgpath = pathsep.join([".", "hw1_sample_images", "sample3.png"])
    img = cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit("Can't open {}".format(imgpath))
    imgpath = pathsep.join([".", "hw1_sample_images", "sample4.png"])
    uninoise = cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
    if uninoise is None:
        sys.exit("Can't open {}".format(imgpath))
    imgpath = pathsep.join([".", "hw1_sample_images", "sample5.png"])
    impnoise = cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
    if impnoise is None:
        sys.exit("Can't open {}".format(imgpath))

    lowpassfilter = np.array([[1., 1., 1., 7., 1., 1., 1.], [1., 1., 1., 7., 1., 1., 1.], [1., 1., 1., 7., 1., 1., 1.], [7., 7., 7., 49., 7., 7., 7.], 
                              [1., 1., 1., 7., 1., 1., 1.], [1., 1., 1., 7., 1., 1., 1.], [1., 1., 1., 7., 1., 1., 1.]]) / 169
    # lowpassfilter2 = np.array([[1.,3.,1.],[3.,9.,3.],[1.,3.,1.]]) / 25
    cleanuni = Conv(uninoise, lowpassfilter)
    cv.imwrite(pathsep.join([".", "result10.png"]), cleanuni)
    print("Uniform noise\nPSNR : {0} -> {1}".format(Psnr(img, uninoise), Psnr(img, cleanuni)))
    # lowpass3x3 = Conv(uninoise, lowpassfilter2)
    # cv.imwrite(pathsep.join([".", "lowpass3x3.png"]), lowpass3x3)
    # print("Uniform noise\nPSNR : {0} -> {1}".format(Psnr(img, uninoise), Psnr(img, lowpass3x3)))
    
    cleanimp = MedianFilter(impnoise, (5, 5))
    cv.imwrite(pathsep.join([".", "result11.png"]), cleanimp)
    print("Impulse noise\nPSNR : {0} -> {1}".format(Psnr(img, impnoise), Psnr(img, cleanimp)))
    # imp3x3 = MedianFilter(impnoise, (3,3))
    # cv.imwrite(pathsep.join([".", "imp3x3.png"]), imp3x3)
    # print("Impulse noise\nPSNR : {0} -> {1}".format(Psnr(img, impnoise), Psnr(img, imp3x3)))

WarnUp()
Prob1()
Prob2()
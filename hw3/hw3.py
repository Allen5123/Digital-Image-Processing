import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import sys
import random
from numba import jit
import math
import time

pathsep = "\\\\"

@jit
def ZeroPadding(img, kernel):
    m, n = img.shape
    s, t = kernel.shape
    #zero padding
    padded = np.zeros((m+(s//2)*2, n+(t//2)*2), dtype=np.uint8)
    padded[s//2:s//2+m, t//2:t//2+n] = img[:,:]
    padded[:s//2,t//2:t//2+n] = padded[-1:-(s//2+1):-1,t//2:t//2+n] = padded[:,:t//2] = padded[:,-1:-(t//2+1):-1] = 0
    return padded

@jit
def Erosion(img, struct):
    padded = ZeroPadding(img, struct)
    m, n = img.shape
    s, t = struct.shape
    ret = np.zeros_like(img, dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            mark = True
            for k in range(-(s//2), s//2+1):
                for l in range(-(t//2), t//2+1):
                    if struct[k+s//2,l+t//2] == 255 and padded[i+k+s//2,j+l+t//2] == 0:
                        mark = False
                        break
                if not mark:
                    break
            if mark:
                ret[i,j] = 255
    return ret

@jit
def Dilation(img, struct):
    structhat = np.fliplr(np.flipud(struct))
    padded = ZeroPadding(img, structhat)
    m, n = img.shape
    s, t = structhat.shape
    ret = np.zeros_like(img, dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            mark = False
            for k in range(-(s//2), s//2+1):
                for l in range(-(t//2), t//2+1):
                    if structhat[k+s//2,l+t//2] == 255 and padded[i+k+s//2,j+l+t//2] == 255:
                        mark = True
                        break
                if mark:
                    break
            if mark:
                ret[i,j] = 255
    return ret              

def BoundaryExtract(img, struct):
    return img - Erosion(img, struct)

@jit
def HoleFilingHelper(img, struct, hole):
    complement = 255 - img
    pre = np.zeros_like(img, dtype=np.uint8)
    pre[hole] = 255
    nxt = Dilation(pre, struct) & complement
    while (not np.array_equal(pre, nxt)):
        pre = nxt
        nxt = Dilation(pre, struct) & complement
    return nxt

def HoleFiling(img, struct, holes):
    for hole in holes:
        img = img | HoleFilingHelper(img, struct, hole)
    return img

@jit
def ConnectedCompHelper(img, struct, start):
    pre = np.zeros_like(img, dtype=np.uint8)
    pre[start] = 255
    nxt = Dilation(pre, struct) & img
    while (not np.array_equal(pre, nxt)):
        pre = nxt
        nxt = Dilation(pre, struct) & img
    # plt.imshow(nxt)
    # plt.show()
    return nxt

def ConnectedComp(img, struct):
    m, n = img.shape
    ret, vis = np.zeros((m, n, 3), dtype=np.uint8), np.zeros((m, n), dtype=np.bool8)
    colors = {(0,0,0):True}
    for i in range(m):
        for j in range(n):
            if not vis[i,j] and img[i,j] > 0:
                newcomp = ConnectedCompHelper(img, struct, (i, j))
                newb, newg, newr = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                while (colors.get((newb, newg, newr)) != None):
                    newb, newg, newr = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                colors[(newb, newg, newr)] = True
                newcolor = np.array((newb, newg, newr))
                for k in range(m):
                    for l in range(n):
                        if newcomp[k,l] == 255:
                            ret[k,l] = newcolor
                            vis[k,l] = True
    return ret

@jit
def Open(img, struct):
    return Dilation(Erosion(img, struct), struct)

def Skeletonizing(img, struct):
    ret, preero = np.zeros_like(img, dtype=np.uint8), img
    nxtero = Erosion(preero, struct)
    while (not np.array_equal(nxtero, np.zeros_like(img, dtype=np.uint8))):
        ret = ret | (nxtero - (Open(nxtero, struct)))
        preero = nxtero
        nxtero = Erosion(preero, struct)
    return ret
                
def Prob1():
    sample1 = cv.imread(pathsep.join(["hw3_sample_images", "sample1.png"]), cv.IMREAD_GRAYSCALE)
    if sample1 is None:
        sys.exit("Can't open {}".format(pathsep.join(["hw3_sample_images", "sample1.png"])))
    result1 = BoundaryExtract(sample1, 255*np.ones((3, 3), dtype=np.uint8))
    cv.imwrite(pathsep.join([".", "result1.png"]), result1)
    holes = [(245,106), (238,134), (282,124), (432,207), (407,269), (500,237), (458,296), (471,298), (490,306)]
    result2 = HoleFiling(sample1, np.array([[0,255,0],[255,255,255],[0,255,0]], dtype=np.uint8), holes)
    cv.imwrite(pathsep.join([".", "result2.png"]), result2)
    result3 = Skeletonizing(sample1, np.array([[0,255,0],[255,255,255],[0,255,0]], dtype=np.uint8))
    cv.imwrite(pathsep.join([".", "result3.png"]), result3)
    result4 = Skeletonizing(255 - sample1, np.array([[0,255,0],[255,255,255],[0,255,0]], dtype=np.uint8))
    cv.imwrite(pathsep.join([".", "result4.png"]), result4)
    result5 = ConnectedComp(sample1, 255*np.ones((3,3), dtype=np.uint8))
    cv.imwrite(pathsep.join([".", "result5.png"]), result5)

Prob1()
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import sys
import random
from numba import jit
import math
import time

pathsep = "/"

@jit
def ZeroPadding(img, ksize):
    m, n = img.shape
    s, t = ksize
    #zero padding
    padded = np.zeros((m+(s//2)*2, n+(t//2)*2), dtype=np.uint8)
    padded[s//2:s//2+m, t//2:t//2+n] = img[:,:]
    padded[:s//2,t//2:t//2+n] = padded[-1:-(s//2+1):-1,t//2:t//2+n] = padded[:,:t//2] = padded[:,-1:-(t//2+1):-1] = 0
    return padded

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

def Conv(img, kernel):
    m, n = img.shape
    s, t = kernel.shape
    ret = np.zeros(img.shape, dtype=np.int32)
    padded = MirrorPadding(img, kernel.shape)
    # print(padded)
    for i in range(m):
        for j in range(n):
            for k in range(-(s//2), s//2+1):
                for l in range(-(t//2), t//2+1):
                    ret[i,j] += padded[s//2+i+k,t//2+j+l] * kernel[s//2+k,t//2+l]
    return ret

@jit
def Erosion(img, struct):
    padded = ZeroPadding(img, struct.shape)
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
    padded = ZeroPadding(img, structhat.shape)
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

@jit
def MeanHelp(img, wsize):
    m, n = img.shape
    s, t = wsize
    mean = np.zeros((m, n), dtype=np.float32)
    padded = MirrorPadding(img, wsize)
    for i in range(m):
        for j in range(n):
            for k in range(-(s//2), s//2+1):
                for l in range(-(t//2), t//2+1):
                    mean[i,j] += padded[i+k+s//2,j+l+t//2]
            mean[i,j] /= s * t
    return mean

def Mean(imgs, wsize):
    ret = []
    for img in imgs:
        ret.append(MeanHelp(img, wsize))   
    return ret

def FeatureSelect(energylist, selectedfeatures):
    energyarray = np.array(energylist)
    m, n = energyarray[0].shape
    features = np.empty((m, n, len(selectedfeatures)), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            features[i,j] = energyarray[selectedfeatures,i,j]
    return features

def LawsMethod(img, energycompute, wsize, selectedfeatures):
    bases = [np.array([[1.,2.,1.]])/6., np.array([[-1.,0.,1.]])/2., np.array([[1.,-2.,1.]])/2.]
    masks = []
    for i in range(len(bases)):
        for j in range(len(bases)):
            masks.append(bases[i].T @ bases[j])
    microstructures = []
    for mask in masks:
        microstructures.append(Conv(img, mask))
    energylist = energycompute(microstructures, wsize)
    # for idx, x in enumerate(energylist):
    #     plt.subplot(3, 3, idx+1)
    #     plt.imshow(x, cmap="gray"), plt.title("{0}".format(idx)), plt.axis('off')
    # plt.show()
    return FeatureSelect(energylist, selectedfeatures)

def UpdateAssignment(datapoints, centers, assignment, normfunc):
    n, k = datapoints.shape[1], centers.shape[1]
    cnt = [0]*k
    loss = 0
    #iterate each datapoint, which is column vector
    for i in range(n):
        minidx, minloss = 0, normfunc(datapoints[:,i] - centers[:,0])
        #iterate each center, which is column vector
        for j in range(k):
            l = normfunc(datapoints[:,i] - centers[:,j])
            if l < minloss:
                minloss, minidx = l, j
        loss += minloss
        cnt[minidx] += 1
        assignment[i] = np.array([1 if idx == minidx else 0 for idx in range(k)])
    # plt.stem([i for i in range(k)], cnt)
    # plt.show()
    return loss

def UpdateCenters(datapoints, centers, assignment, normfunc):
    k = centers.shape[1]
    loss = 0
    for i in range(k):
        clustermembers = np.sum(assignment[:,i])
        if clustermembers > 0:
            centers[:,i] = (datapoints @ assignment[:,i]) / clustermembers
            for j in range(datapoints.shape[1]):
                loss += assignment[j,i] * normfunc(datapoints[:,j] - centers[:,i])
    return loss

@jit
def RandomAssign(n, k):
    ret = np.zeros((n, k), dtype=np.int8)
    for i in range(n):
        ret[i,np.random.randint(0,k)] = 1
    return ret

def Kmeans(datapoints, k, normfunc):
    # f features, n data, k clusters
    f, n = datapoints.shape
    loss = []
    assignment = RandomAssign(n, k)
    centers = np.empty((f, k))
    # print(datapoints.shape, centers.shape, assignment.shape)
    loss.append(UpdateCenters(datapoints, centers, assignment, normfunc))
    while (True):
        loss.append(UpdateAssignment(datapoints, centers, assignment, normfunc))
        if (abs(loss[-1] - loss[-2]) / loss[-1] <= 5.e-10):
            break
        loss.append(UpdateCenters(datapoints, centers, assignment, normfunc))
        if (abs(loss[-1] - loss[-2]) / loss[-1] <= 5.e-10):
            break
    # plt.plot(np.arange(len(loss)), loss)
    # plt.show()
    return assignment, centers

@jit
def OneNorm(x):
    return np.sum(np.abs(x))

@jit
def Bfs(img, position, vis, fillcolor):
    dirs = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    cnt = 0
    m, n, __ = img.shape
    color = img[position].copy()
    queue = [position]
    retvis = vis.copy()
    retvis[position] = True
    while len(queue) > 0:
        r, c = queue.pop(0)
        img[r,c] = fillcolor
        cnt += 1
        for dir in dirs:
            nxtr, nxtc = r + dir[0], c + dir[1]
            if nxtr >= 0 and nxtr < m and nxtc >= 0 and nxtc < n and not retvis[nxtr,nxtc] and np.array_equal(img[nxtr,nxtc], color):
                queue.append((nxtr, nxtc))
                retvis[nxtr,nxtc] = True
    return cnt, retvis

def Improve(img, holesize, featurevectors, centers, colors, normfunc):
    m, n, __ = img.shape
    vis = np.zeros((m, n), np.bool8)
    for i in range(m):
        for j in range(n):
            if not vis[i,j]:
                compsize, newvis = Bfs(img, (i,j), vis, img[i,j])
                if compsize <= holesize:
                    dist = []
                    for k in range(centers.shape[1]):
                        dist.append((k, normfunc(featurevectors[i,j] - centers[:,k])))
                    dist.sort(key = lambda x : x[1])
                    __, newvis = Bfs(img, (i,j), vis, colors[dist[1][0]])
                    # plt.imshow(img)
                    # plt.show()
                else:
                    print(i, j, compsize)
                vis = newvis
    return img

def Prob2():
    sample2 = cv.imread(pathsep.join(["hw3_sample_images", "sample2.png"]), cv.IMREAD_GRAYSCALE)
    m, n = sample2.shape
    if sample2 is None:
        sys.exit("Can't open {}".format(pathsep.join(["hw3_sample_images", "sample2.png"])))
    # featurevectors = LawsMethod(sample2, Mean, (15, 15), [0,1,3,4,6,7])
    # featurevectors = LawsMethod(sample2, Mean, (15, 15), [0,2,5])
    featurevectors = LawsMethod(sample2, Mean, (15, 15), [0,2,6])
    numofcluster = 4
    assignment, centers = Kmeans(featurevectors.reshape((m * n, featurevectors.shape[2])).T, numofcluster, OneNorm)
    assignment = assignment.reshape((m, n, numofcluster))
    colors = np.array([[np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)] for i in range(numofcluster)])
    result6 = np.empty((m, n, 3), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            for k in range(numofcluster):
                if assignment[i,j,k] == 1:
                    result6[i,j] = colors[k]
                    break
    cv.imwrite(pathsep.join([".", "result6.png"]), result6)
    result7 = Improve(result6.copy(), 970, featurevectors, centers, colors, OneNorm)
    cv.imwrite(pathsep.join([".", "result7.png"]), result7)

Prob1()
Prob2()
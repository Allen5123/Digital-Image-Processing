import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import sys
import math
import time

pathsep = "/"

def Padding(img, kernel):
    m, n = img.shape
    s, t = kernel.shape
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
    padded = Padding(img, kernel)
    # print(padded)
    for i in range(m):
        for j in range(n):
            for k in range(-(s//2), s//2+1):
                for l in range(-(t//2), t//2+1):
                    ret[i,j] += padded[s//2+i+k,t//2+j+l] * kernel[s//2+k,t//2+l]
    return ret

def Compress(img):
    res = np.empty(img.shape)
    m, n = img.shape
    mn = mx = img[0,0]
    for i in range(m):
        for j in range(n):
            mn = min(mn, img[i,j])
            mx = max(mx, img[i,j])
    for i in range(m):
        for j in range(n):
            res[i,j] = (img[i,j] - mn) * 255 // (mx - mn)
    return res.astype(np.uint8)

def Gradx(img):
    sobx = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
    return Conv(img, sobx)

def Grady(img):
    soby = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).T
    return Conv(img, soby) 

def Sobel(img, threshold):
    gradx, grady = Gradx(img), Grady(img)
    grad = (gradx**2 + grady**2) ** 0.5
    ret = np.empty_like(grad)
    m, n = img.shape
    for i in range(m):
        for j in range(n):
            if grad[i,j] >= threshold:
                ret[i,j] = 255
            else:
                ret[i,j] = 0
    # plt.figure(1)
    # plt.subplot(111)
    # plt.imshow(grad, cmap="jet")
    # plt.colorbar(), plt.axis("off")
    # plt.show()
    return ret.astype(np.uint8), grad

def Gaussian(sz):
    if sz % 2 == 0:
        sz += 1
    gauss = np.empty((sz, sz))
    sigma = sz / 6
    summ = 0
    for i in range(sz):
        for j in range(sz):
            gauss[i,j] = math.exp(-((i-sz//2)**2+(j-sz//2)**2) / (2*(sigma**2)))        
            summ += gauss[i,j]
    gauss /= summ
    # x, y = np.meshgrid(np.arange(sz), np.arange(sz))
    # ax = plt.subplot(111, projection='3d')
    # ax.plot_surface(x, y, gauss, cmap='jet')
    # plt.show()
    return gauss
    
def Canny(img, gsz, TL, TH):
    #smooth with Gaussian
    smooth = Conv(img, Gaussian(gsz))
    #Compute gradient and angle
    gradx, grady = Gradx(smooth), Grady(smooth)
    grad = (gradx**2 + grady**2) ** 0.5
    angle = np.arctan2(grady, gradx) * (180 / np.pi) #degree
    # plt.subplot(111)
    # plt.imshow(grad, cmap="jet")
    # plt.colorbar(), plt.axis("off")
    # plt.show()
    #nonmaxima suppression
    m, n = grad.shape
    gn = np.empty_like(grad)
    for i in range(m):
        for j in range(n):
            dx1 = dy1 = dx2 = dy2 = 0
            #horizontal edge
            if (angle[i,j] >= -157.5 and angle[i,j] < 157.5) or (angle[i,j] <= 22.5 and angle[i,j] > -22.5):
                dx1, dy1, dx2, dy2 = 0, -1, 0, 1
            #vertical edge
            elif (angle[i,j] <= -67.5 and angle[i,j] > -112.5) or (angle[i,j] <= 112.5 and angle[i,j] > 67.5):
                dx1, dy1, dx2, dy2 = -1, 0, 1, 0
            #+45 degree edge
            elif (angle[i,j] <= 157.5 and angle[i,j] > 112.5) or (angle[i,j] <= -22.5 and angle[i,j] > -67.5):
                dx1, dy1, dx2, dy2 = -1, -1, 1, 1
            #-45 degree edge
            else:
                dx1, dy1, dx2, dy2 = 1, -1, -1, 1
            if (i+dx1 < 0 or i+dx1 >= m or j+dy1 < 0 or j+dy1 >= n or grad[i,j] >= grad[i+dx1,j+dy1]) and \
               (i+dx2 < 0 or i+dx2 >= m or j+dy2 < 0 or j+dy2 >= n or grad[i,j] >= grad[i+dx2,j+dy2]):
                gn[i,j] = grad[i,j]
            else:
                gn[i,j] = 0
    #hysteresis thresholding
    hh = np.empty((m, n), dtype=np.uint8)
    mark = np.full((m, n), False, dtype=np.bool8)
    edgepoints = []
    for i in range(m):
        for j in range(n):
            if gn[i,j] >= TH:
                hh[i,j] = 255
                edgepoints.append((i,j))
                mark[i,j] = True
            elif gn[i,j] >= TL:
                hh[i,j] = 100
            else:
                hh[i,j] = 0
    #Connected component labeling method
    dirs = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    ret = hh.copy()
    while len(edgepoints) > 0:
        i, j = edgepoints[-1]
        edgepoints.pop()
        for d in dirs:
            ii, jj = i + d[0], j + d[1]
            if ii >= 0 and ii < m and jj >= 0 and jj < n and hh[ii,jj] > 0:
                ret[ii,jj] = 255
                if not mark[ii,jj]:
                    edgepoints.append((ii,jj))
                    mark[ii,jj] = True
    for i in range(m):
        for j in range(n):
            if ret[i,j] < 255:
                ret[i,j] = 0

    # plt.figure("Non-maximal suppression TL:{0}, TH:{1}".format(TL, TH))
    # plt.subplot(111), plt.axis('off')
    # plt.imshow(gn, cmap="gray")
    # plt.figure("Hysteretic thresholding TL:{0}, TH:{1}".format(TL, TH))
    # plt.subplot(111), plt.axis('off')
    # plt.imshow(hh, cmap="gray")
    # plt.figure("Connected component labeling method TL:{0}, TH:{1}".format(TL, TH))
    # plt.subplot(111), plt.axis('off')
    # plt.imshow(ret, cmap="gray")
    # plt.show()
    return ret            

def Laplacian(img):
    lap = np.array([[1.,1.,1.],[1.,-8.,1.],[1.,1.,1.]])
    return Conv(img, lap)

def LoG(img, gsz, threshold):
    smooth = Conv(img, Gaussian(gsz))
    lap = -Laplacian(smooth)
    # plt.subplot(111)
    # plt.imshow(lap, cmap="jet")
    # plt.colorbar(), plt.axis("off")
    # plt.show()
    ret = np.empty_like(img)
    m, n = img.shape
    for i in range(m):
        for j in range(n):
            #horizontal, vertical, +45 degree, -45 degree
            if (j-1>=0 and j+1<n and lap[i,j-1]*lap[i,j+1]<0 and abs(lap[i,j-1]-lap[i,j+1])>=threshold) or \
               (i-1>=0 and i+1<m and lap[i-1,j]*lap[i+1,j]<0 and abs(lap[i-1,j]-lap[i+1,j])>=threshold) or \
               (i-1>=0 and j-1>=0 and i+1<m and j+1<n and lap[i-1,j-1]*lap[i+1,j+1]<0 and abs(lap[i-1,j-1]-lap[i+1,j+1])>=threshold) or \
               (i-1>=0 and j+1<n and i+1<m and j-1>=0 and lap[i+1,j-1]*lap[i-1,j+1]<0 and abs(lap[i+1,j-1]-lap[i-1,j+1])>=threshold):
                   ret[i,j] = 255
            else:
                ret[i,j] = 0
            # mx = mn = lap[i,j]
            # for k in range(-1,2):
            #     for l in range(-1,2):
            #         if i+k>=0 and i+k < m and j+l>=0 and j+l<n:
            #             mx = max(mx, lap[i+k,j+l])
            #             mn = min(mn, lap[i+k,j+l])
            # if mx - mn >= threshhold:
            #     ret[i,j] = 255
            # else:
            #     ret[i,j] = 0
    return ret

def Unsharp(img, gsz, c):
    filted = Conv(img, Gaussian(gsz))
    m, n = img.shape
    ret = np.empty((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            val = c / (2 * c - 1) * img[i,j] - (1 - c) / (2 * c - 1) * filted[i,j]
            if val < 0:
                ret[i,j] = 0
            elif val < 255:
                ret[i,j] = val
            else:
                ret[i,j] = 255
    # diff = img.astype(np.int32) - ret.astype(np.int32)
    # plt.subplot(111)
    # plt.imshow(diff)
    # plt.colorbar()
    # plt.show()
    return ret.astype(np.uint8)

#image coordinate to Cartesian
def ItoC(coor, imgsz):
    j, k = coor
    m, n = imgsz
    return  ((k+1)-1/2, m+1/2-(j+1))

#Cartesian to image coordinate
def CtoI(coor, imgsz):
    x, y = coor
    m, n = imgsz
    return (m+1/2-y-1, x-1+1/2)

#Bilinear interpolation using image coordinate
def BilinIntp(coor, img):
    m, n = img.shape
    i, j = coor
    p, q = math.floor(i), math.floor(j)
    a, b = i - p, j - q
    if p < 0 or p >= m or q < 0 or q >= n:
        return 0
    ret = (1-a)*(1-b)*img[p,q]
    if q+1 < img.shape[1]:
        ret += (1-a)*b*img[p,q+1]
    if p+1 < img.shape[0]:
        ret += a*(1-b)*img[p+1,q]
    if p+1 < img.shape[0] and q+1 < img.shape[1]:
        ret += a*b*img[p+1,q+1]
    return ret
    
def Translation(tx, ty):
    return np.array([[1., 0., tx], [0., 1., ty], [0., 0., 1.]])

def Scaling(sx, sy):
    return np.array([[sx, 0., 0.], [0., sy, 0.], [0., 0., 1.]])

def Rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])

#Transform : (u,v) |-> (x,y)
def Transform(img, operations):
    transform = np.identity(3, dtype=np.float32)
    for operation in operations:
        transform = transform @ operation
    invtrans = np.linalg.inv(transform)
    ret = np.empty_like(img)
    m, n = img.shape
    #reverse address
    #      ItoC       Inverse      CtoI
    #(x,y) -> (cx, cy) -> (cu, cv) -> (u,v)
    for x in range(m):
        for y in range(n):
            cx, cy = ItoC((x, y), (m, n))
            cu, cv, __ = tuple(np.squeeze(invtrans @ np.array([[cx, cy, 1]]).T))
            ret[x,y] = BilinIntp(CtoI((cu, cv), (m, n)), img)
    # plt.subplot(111)
    # plt.imshow(ret, cmap="gray"), plt.axis("off")
    # plt.show()
    return ret

def Hough(oriimg, edgeimg):
    m, n = edgeimg.shape
    mxx, mxy = ItoC((0, n - 1), (m, n))
    mxrho = math.ceil((mxx**2 + mxy**2)**0.5)
    mntheta, mxtheta = -90, 180
    ret = np.zeros((2 * mxrho + 1, mxtheta - mntheta + 1), dtype=np.uint32)
    for i in range(m):
        for j in range(n):
            x, y = ItoC((i, j), (m, n))
            phi = math.atan2(y, x)
            if edgeimg[i,j] > 0:
                for theta in range(mntheta, mxtheta + 1):
                    rho = round(x*np.cos(np.radians(theta)) + y*np.sin(np.radians(theta)))
                    if theta >= np.rad2deg(phi-np.pi/2) and theta <= np.rad2deg(phi+np.pi/2):
                        ret[rho+mxrho,theta-mntheta] += 1
                            
    # plt.figure(1)
    # ax = plt.subplot(111)
    # plt.imshow(ret, cmap="jet", extent = [mntheta, mxtheta, mxrho, -mxrho])
    # plt.colorbar(), ax.set_aspect('auto'), plt.xlabel(r"$\theta (\degree)$"), plt.ylabel(r"$\rho$")
    # plt.show()
        
    lines = []
    for i in range(-mxrho, mxrho + 1):
        for j in range(mntheta, mxtheta + 1):
            if ret[i+mxrho,j-mntheta] > 150.:
                lines.append((i,j))
    def Cmp(a):
        return ret[a[0]+mxrho, a[1]-mntheta]
    lines.sort(key = Cmp, reverse=True)

    def PlotLine(img, lines):
        m, n = img.shape
        plt.imshow(img, cmap="gray")
        for i, line in enumerate(lines):
            rho, theta = line
            if rho == 0:
                continue
            print(rho, theta)
            x0, y0 = rho*math.cos(math.radians(theta)), rho*math.sin(math.radians(theta))
            #parameter eq. x=x0-y0*t, y=y0+x0*t, t in [-inf,inf]
            x, y = [], []
            if theta == 0 or theta == 180:
                x.append(x0), x.append(x0)
                y.append(0), y.append(m-1)
            elif theta == 90:
                x.append(0), x.append(n-1)
                y.append(y0), y.append(y0)
            else:
                if y0+x0*x0/y0 >= 0 and y0+x0*x0/y0 < m:
                    x.append(0), y.append(y0+x0*x0/y0)
                if x0-y0*(-y0/x0) >= 0 and x0-y0*(-y0/x0) < n:
                    x.append(x0-y0*(-y0/x0)), y.append(0)
                if y0+x0*(x0-(n-1))/y0 >= 0 and y0+x0*(x0-(n-1))/y0 < m:
                    x.append(n-1), y.append(y0+x0*(x0-(n-1))/y0)
                if x0-y0*(m-1-y0)/x0 >= 0 and x0-y0*(m-1-y0)/x0 < n:
                    x.append(x0-y0*(m-1-y0)/x0), y.append(m-1)
            #flipp y
            y[0], y[1] = (m - 1) - y[0], (m - 1) - y[1]
            plt.plot(x, y)
        plt.axis("off")
        plt.show()

    # PlotLine(oriimg, lines[:9])
    return ret
    
def Prob1():
    sample1 = cv.imread(pathsep.join([".", "hw2_sample_images", "sample1.png"]), cv.IMREAD_GRAYSCALE)
    sample2 = cv.imread(pathsep.join([".", "hw2_sample_images", "sample2.png"]), cv.IMREAD_GRAYSCALE)
    if sample1 is None:
        sys.exit("Can't open {}".format(pathsep.join([".", "hw2_sample_images", "sample1.png"])))
    if sample2 is None:
        sys.exit("Can't open {}".format(pathsep.join([".", "hw2_sample_images", "sample2.png"])))

    #Sobel
    res2, res1 = Sobel(sample1, 200)
    cv.imwrite(pathsep.join([".", "result1.png"]), Compress(res1))
    cv.imwrite(pathsep.join([".", "result2.png"]), res2)
    
    #Canny
    res3 = Canny(sample1, 5, 55., 182.)
    cv.imwrite(pathsep.join([".", "result3.png"]), res3)   

    #Laplacian of Gaussian
    res4 = LoG(sample1, 7, 88.)
    cv.imwrite(pathsep.join([".", "result4.png"]), res4)

    #Unsharp
    res5 = Unsharp(sample2, 5, 18./30.)
    cv.imwrite(pathsep.join([".", "result5.png"]), res5)
    
    #Hough Transform
    res6 = Hough(sample1, res3)
    cv.imwrite(pathsep.join([".", "result6.png"]), Compress(res6))

def SineWrap(img, culist, cvlist, cxlist, cylist, wx, wy, phix, phiy):
    controlnum = len(culist)
    m, n = img.shape
    A = np.empty((controlnum, 5)) #[[1, x, y, sin(w_x*(x+phi)),sin(w_y*(y+phi)]]
    for i, (cx, cy) in enumerate(zip(cxlist, cylist)):
        A[i,0], A[i,1], A[i,2], A[i,3], A[i,4] = 1, cx, cy, math.sin(wx*(cx+phix)), math.sin(wy*(cy+phiy))    
    pseudoinvA = np.linalg.pinv(A)
    a, b = np.squeeze(pseudoinvA @ np.expand_dims(culist, axis=0).T), \
           np.squeeze(pseudoinvA @ np.expand_dims(cvlist, axis=0).T)
    ret = np.zeros_like(img)
    #reverse address
    for x in range(m):
        for y in range(n):
            cx, cy = ItoC((x, y), (m, n))
            cu, cv = a[0]+a[1]*cx+a[2]*cy+a[3]*math.sin(wx*(cx+phix))+a[4]*math.sin(wy*(cy+phiy)), \
                     b[0]+b[1]*cx+b[2]*cy+b[3]*math.sin(wx*(cx+phix))+b[4]*math.sin(wy*(cy+phiy)),
            ret[x,y] = BilinIntp(CtoI((cu, cv), (m, n)), img)
    # plt.subplot()
    # plt.imshow(ret, cmap="gray"), plt.axis("off")
    # plt.show()
    return ret
        
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

def Improve(img):
    edge, __ = Sobel(img, 600)
    ret = img.copy()
    m, n = img.shape
    for i in range(m):
        for j in range(n):
            if edge[i,j] == 255:
                ret[i,j] = 0         
    # plt.subplot(131), plt.title("Median"), plt.axis("off")
    # plt.imshow(MedianFilter(img, (3, 3)), cmap="gray")
    # plt.subplot(132), plt.title("Sobel"), plt.axis("off")
    # plt.imshow(ret, cmap="gray")
    ret = MedianFilter(ret, (3, 3))
    # plt.subplot(133), plt.title("Sobel and Median"), plt.axis("off")
    # plt.imshow(ret, cmap="gray")
    # plt.show()
    return ret

def Prob2():
    sample3 = cv.imread(pathsep.join([".", "hw2_sample_images", "sample3.png"]), cv.IMREAD_GRAYSCALE)
    sample5 = cv.imread(pathsep.join([".", "hw2_sample_images", "sample5.png"]), cv.IMREAD_GRAYSCALE)
    if sample3 is None:
        sys.exit("Can't open {}".format(pathsep.join([".", "hw2_sample_images", "sample3.png"])))
    if sample5 is None:
        sys.exit("Can't open {}".format(pathsep.join([".", "hw2_sample_images", "sample5.png"])))
    
    imp = Improve(sample3)
    cv.imwrite(pathsep.join([".", "sample3improve.png"]), imp)
    top = Transform(sample3, [Translation(150, 300), Scaling(0.5, 0.5)])
    bottom = Transform(top, [Translation(300, 300), Rotation(np.radians(182)), Translation(-300, -300)])
    left = Transform(top, [Translation(300, 300), Rotation(np.radians(92)), Translation(-300, -300)])
    right = Transform(top, [Translation(300, 300), Rotation(np.radians(272)), Translation(-300, -300)])
    res7 = top | bottom | left | right
    cv.imwrite(pathsep.join([".", "result7.png"]), res7)
    
    # sample6 = cv.imread(pathsep.join([".", "hw2_sample_images", "sample6.png"]), cv.IMREAD_GRAYSCALE)
    # plt.subplot(111), plt.imshow(sample6, cmap="gray")
    # plt.plot([0,sample6.shape[1]-1],[55,55])
    # plt.plot([0,sample6.shape[1]-1],[315,315])
    # plt.plot([0,sample6.shape[1]-1],[510,510])
    # plt.plot([42,42], [0,sample6.shape[0]-1])
    # plt.plot([564,564], [0,sample6.shape[0]-1])
    # line = np.linspace(0,sample6.shape[1]-1,1000)
    # plt.plot(line, 55+30*np.sin(2*np.pi/149*(line+36.)), lw=1.5)
    # plt.plot(line, 315+30*np.sin(2*np.pi/149*(line+36.)), lw=1.5)
    # plt.plot(line, 510+30*np.sin(2*np.pi/149*(line+36.)), lw=1.5)
    # plt.plot(42.+20.*np.sin(2*np.pi/149*(line-75)), line, lw=1.5)
    # plt.plot(564.+20.*np.sin(2*np.pi/149*(line-75)), line, lw=1.5)
    # plt.show()

    x0, y0, wx, wy, phix, phiy = 30., 20., 2*math.pi/149., 2*math.pi/149., 36., -75.
    top, mid, bottom, left, right = (sample5.shape[0]-55), (sample5.shape[0]-315), (sample5.shape[0]-510), 42, 564
    line = np.linspace(0, sample5.shape[1]-1, 10)
    ones = np.ones(len(line))
    culist = np.concatenate((line, line, line, left*ones, right*ones))
    cvlist = np.concatenate((top*ones, mid*ones, bottom*ones, line, line))
    cxlist = np.concatenate((line, line, line, left+y0*np.sin(wy*(line+phiy)), right+y0*np.sin(wy*(line+phiy))))
    cylist = np.concatenate((top+x0*np.sin(wx*(line+phix)), mid+x0*np.sin(wx*(line+phix)), bottom+x0*np.sin(wx*(line+phix)), line, line))
    res8 = SineWrap(sample5, culist, cvlist, cxlist, cylist, wx, wy, phix, phiy)
    cv.imwrite(pathsep.join([".", "result8.png"]), res8)

Prob1()
Prob2()
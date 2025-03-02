import numpy as np
import cv2

# Average Gradient (reflects the change of detail information in the image, more clarity means higher average gradient)
def avgerage_gradient(image_path):
    im = cv2.imread(image_path)
    if im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    avg_grad = 0
    for i in range(im.shape[0]-1):
        for j in range(im.shape[1]-1):
            avg_grad += np.sqrt(((im[i+1,j]-im[i,j])**2+(im[i,j+1]-im[i,j])**2)/2)
    
    avg_grad = avg_grad/((im.shape[0]-1)*(im.shape[1]-1))
    return avg_grad
            

# Information Entropy (larger IE value means more richer color information)
def information_entropy(image_path):
    im = cv2.imread(image_path)
    if im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    hist = cv2.calcHist([im], [0], None, [256], [0, 256])
    hist = hist.ravel()/hist.sum()

    ie = 0
    for i in range(hist.shape[0]):
        ie += hist[i]*np.log2(hist[i]+1e-10) 
    
    ie = -ie

    return ie


# Edge Gradient (large EG value larger edge retention)
def edge_gradient(image_path):
    im = cv2.imread(image_path)
    if im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    

# underwater color image quality evaluation index (UCIQE)
"""
ref: M. Yang and A. Sowmya, "An Underwater Color Image Quality Evaluation Metric," 
in IEEE Transactions on Image Processing, vol. 24, no. 12, pp. 6062-6071, Dec. 2015, 
doi: 10.1109/TIP.2015.2491020.
"""
def UCIQE(image_path, c1=0.4680, c2=0.2745, c3=0.2576):
    im = cv2.imread(image_path)

    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(im_lab)

    chroma = np.sqrt(A**2 + B**2)
    chroma_mean = np.mean(chroma.flatten())+1e-10
    chroma_std = np.sqrt(np.mean(chroma.flatten()**2 - chroma_mean**2))

    saturation = chroma/(np.mean(L)+1e-10)
    saturtion_mean = np.mean(saturation.flatten())

    contrast = np.max(L.flatten()) - np.min(L.flatten())

    uciqe = c1*chroma_std + c2*contrast + c3*saturtion_mean

    return uciqe

# underwater image color metric index (UIQM)
"""
ref: K. Panetta, C. Gao and S. Agaian, "Human-Visual-System-Inspired Underwater Image Quality Measures,"
in IEEE Journal of Oceanic Engineering, vol. 41, no. 3, pp. 541-551, July 2016,
doi: 10.1109/JOE.2015.2469915

"""
def UIQM(image_path, p1=0.0282, p2=0.2953, p3=3.5753):
    im = cv2.imread(image_path) #bgr
    r = im[:,:,2]
    g = im[:,:,1]
    b = im[:,:,0]
    
    RG = r - g
    YB = (r + g)/2 - b

    rgl = np.sort(RG.flatten())
    ybl = np.sort(YB.flatten())

    T1 = int(0.1*rgl.shape[0])
    T2 = int(0.1*ybl.shape[0])

    rgl__tr = rgl[T1:-T2]
    ybl__tr = ybl[T1:-T2]

    rg_mean = np.mean(rgl__tr) + 1e-10
    rg_var = np.mean((rgl__tr - rg_mean)**2)
    yb_mean = np.mean(ybl__tr) + 1e-10
    yb_var = np.mean((ybl__tr - yb_mean)**2)

    uicm = -0.0268*np.sqrt(rg_mean**2 + yb_mean**2) + 0.1586*np.sqrt(rg_var + yb_var)
    
    Rsobel = cv2.Sobel(im[:,:,2], cv2.CV_64F, 1, 1, ksize=3)
    Gsobel = cv2.Sobel(im[:,:,1], cv2.CV_64F, 1, 1, ksize=3)
    Bsobel = cv2.Sobel(im[:,:,0], cv2.CV_64F, 1, 1, ksize=3)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299*Reme + 0.587*Geme + 0.114*Beme

    uiconm = logamee(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

    uiqm = p1*uicm + p2*uism + p3*uiconm

    return uiqm

def eme(ch,blocksize=8):

    num_x = int(np.ceil(ch.shape[0] / blocksize))
    num_y = int(np.ceil(ch.shape[1] / blocksize))
    
    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]

            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0: blockmin+=1
            if blockmax == 0: blockmax+=1
            eme += w * np.log(abs(blockmax / blockmin)+1e-10)
            
    return eme

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch,blocksize=8):

    num_x = np.ceil(ch.shape[0] / blocksize)
    num_y = np.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(int(num_x)):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(int(num_y)):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]
            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            top = plipsub(blockmax,blockmin)
            bottom = plipsum(blockmax,blockmin)

            m = top/bottom
            if m ==0.:
                s+=0
            else:
                s += (m) * np.log(m)

    return plipmult(w,s)

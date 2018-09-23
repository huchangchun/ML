#-*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
#from skleran import datasets
import os
from skimage import io
def getImgAsMat(index):
    ds = datasets.fetch_olivetti_faces()
    return np.mat(ds.images[index])

def getImgAsMatFromFile(filename):
    img = io.imread(filename,as_grey=True)
    return np.mat(img)

def plotImg(imgMat):
    plt.imshow(imgMat,cmap = plt.cm.gray)
    plt.show()
    
def recoverBySVD(imgMat,k):
    U,s,V = la.svd(imgMat)
    #chooose top k important singular values (or eigens)
    Uk = U[:,0:k]
    Sk = np.diag(s[0:k])
    Vk = V[0:k,:]
    #recover the image
    imgMat_new = Uk * Sk * Vk
    return imgMat_new
if __name__=="__main__":
   
     
         
    file_path =r'C:\Program Files (x86)\Tesseract-OCR\data'
    fns = [os.path.join(root, fn) for root, dirs, files in os.walk(file_path) for fn in files]
    for fn in fns:
        fpath, fname = os.path.split(fn)
        filepath = os.path.join(fpath, fname)    
        A = getImgAsMatFromFile(filepath)
        imageNew = recoverBySVD(A,90)
        newfile = os.path.join(fpath,"new"+fname.replace("jpg","png"))
        plotImg(imageNew)
        #io.imsave(newfile,imageNew,)
   
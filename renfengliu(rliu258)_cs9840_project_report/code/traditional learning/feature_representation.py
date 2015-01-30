import time

import numpy as np
import cv2
import cv2.cv as cv
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from dataset import *

def get_frontal_face(img, img_size, scale=1.1, size=5, ):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.1, 5)
    if len(faces) > 0:
        (x,y,w,h) = faces[0]
        sub_img = img[x:x+w+4, y:y+h+4]
        sub_img = cv2.resize(sub_img, (img_size, img_size))
        return sub_img
    return None


def build_gabor_filters(img_size):
    filters = []
    ksize = img_size
    lambd =[1.17, 1.65, 2.33, 3.30, 4.67]
    for theta in np.arange(0, np.pi, np.pi / 16):
        for l in lambd:
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, l, 0.5, 0, ktype=cv2.CV_32F)
#             kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters

def apply_gabor_process(img, filters):
    result = []
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        l = np.ravel(fimg)
        result.extend(l.tolist())
    return result

def filter_by_frontal_faces(x, y, img_size):
    print "Begin to get frontal faces using OpenCV..."
    xx = []
    yy = []
    for i in range(x.shape[0]):
        img = x[i].reshape(48, 48)
        sub = get_frontal_face(img, img_size)
        if sub != None:
            xx.append(np.ravel(sub))
            yy.append(y[i])
    xxx = np.zeros((len(xx), len(xx[0])))
    yyy = np.zeros((len(xx), 1))

    for i in range(len(xx)):
        xxx[i] = np.array(xx[i])
        yyy[i] = int(yy[i])

    print "Finished getting frontal faces."
    print "data size is : ", xxx.shape[0]
    print "Positive labels: ", np.count_nonzero(yyy)
    print "Negtive labels: ", (xxx.shape[0] - np.count_nonzero(yyy))
    # print np.count_nonzero(yyy)

    return xxx, yyy

def apply_gabor_filters(x, img_size, filters):
    img = x[0].reshape(img_size, img_size)
    res1 = apply_gabor_process(img, filters)
    f = len(res1)
    xx = np.zeros((x.shape[0], f))
    for i in range(x.shape[0]):
        img = x[i].reshape(img_size, img_size)
        xx[i] = apply_gabor_process(img, filters)
        xx[i] =  xx[i].reshape(-1)
    return xx

def apply_histogram_equalization(x):
    new_x = np.zeros_like(x)
    for i in range(x.shape[0]):
        img = x[i].reshape(48, 48)
        img = img.astype('uint8')
        equ = cv2.equalizeHist(img)
        new_x[i] = equ.reshape(-1)
    return new_x

def scale_image(x, dst_img_size):
    img_num = x.shape[0]
    new_x = np.ndarray((x.shape[0], dst_img_size*dst_img_size), dtype='uint8')
    for i in range(img_num):
        img = x[i]
        img = img.reshape(48, 48)
        img = cv2.resize(img, (dst_img_size, dst_img_size))
        new_x[i] = img.reshape(-1)
    return new_x

def get_pixels_diff(x):
    num = x.shape[1] * (x.shape[1] -1)
    # print x.shape[0], num
    num_of_pixels = num *x.shape[0]
    percent = int(num_of_pixels/10)
    print "number of pixels is :", num_of_pixels
    new_x = np.ndarray((x.shape[0], num), dtype='int8')
    cnt = 0
    for i in range (x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[1]-1):
                new_x[i][j*48 + k] = x[i][j] - x[i][k+1]
                cnt = cnt + 1
                if (cnt % percent) == 0:
                	print "processed %.1f%% "%(float(100*cnt)/num_of_pixels)

    return new_x

def get_img_pca(x):
    new_x=np.ndarray((x.shape[0], 20*48) )
    pca = PCA(n_components=20)
    for i in range(x.shape[0]):
        img = x[i]
        results = pca.fit_transform(img.reshape(48,48))
        new_x[i] = results.reshape(1, results.shape[0]*results.shape[1])
#         r[i] = results.components_.reshape(1,results.components_.shape[0]*results.components_.shape[1])
    return new_x

def get_img_hog(img):
    bin_n = 16
    img = img.astype('f4')
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    
    bins = np.int32(bin_n*ang/(2*np.pi))

    bin_cells = bins[:16,:16], bins[16:32, :16], bins[32:, :16],\
                bins[:16,16:32], bins[16:32, 16:32], bins[32:, 16:32],\
                bins[:16,32:], bins[16:32, 32:], bins[32:, 32:]
    
    mag_cells = mag[:16,:16], mag[16:32, :16], mag[32:, :16],\
                mag[:16,16:32], mag[16:32, 16:32], mag[32:, 16:32],\
                mag[:16,32:], mag[16:32, 32:], mag[32:, 32:]        
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist
    
def get_hog_features(x):
    new_x = np.ndarray((x.shape[0], 144))
    for i in range(x.shape[0]):
        img = x[i].reshape(48, 48)
        tmp = get_img_hog(img)
        new_x[i] = tmp.reshape(-1)
    return new_x

def get_mouth_area(x):
    new_x = np.ndarray((x.shape[0],14*24))
    for i in range(x.shape[0]):
        img = x[i].reshape(48, 48)
        img = img[32:46, 12:36]
        new_x[i] = img.reshape(-1)

    return new_x

def get_sift_features(x):
    num_of_key_points = 15

    sift = cv2.SIFT(num_of_key_points)
    new_x = np.ndarray((x.shape[0], num_of_key_points*128))
    for i in range(x.shape[0]):
        img = x[i]
        img = img.reshape(48, 48)
        kp, des = sift.detectAndCompute(img,None)
        des = des.reshape(-1)
        if len(des)<num_of_key_points*128:
            remain= num_of_key_points*128 - len(des)
    #         print remain
            r = [0]*remain
            des = np.append(des, r)
        else:
            des = des[:num_of_key_points*128]
        new_x[i] =des

    return new_x

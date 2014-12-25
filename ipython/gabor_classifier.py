import time

import numpy as np
import cv2
import cv2.cv as cv
import matplotlib.cm as cm
from smile import load_smile_data
import matplotlib.pyplot as plt

from sklearn import svm, metrics
from sklearn import cross_validation

def get_frontal_face(img, img_size, scale=1.1, size=5, ):
    faces = face_cascade.detectMultiScale(img, 1.1, 5)
    if len(faces) > 0:
        (x,y,w,h) = faces[0]
        sub_img = img[x:x+w+4, y:y+h+4]
        sub_img = cv2.resize(sub_img, (img_size, img_size))
        return sub_img
    return None


def build_filters(img_size):
    filters = []
    ksize = img_size
    lambd =[1.17, 1.65, 2.33, 3.30, 4.67]
    for theta in np.arange(0, np.pi, np.pi / 16):
        for l in lambd:
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, l, 0.5, 0, ktype=cv2.CV_32F)
#             kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters

def process(img, filters):
    result = []
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        l = np.ravel(fimg)
        result.extend(l.tolist())
    return result

def filter_by_frontal_faces(x, y, img_size):
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

	print "data size is : ", xxx.shape[0]
	print "Positive labels: ", np.count_nonzero(yyy)
	print "Negtive labels: ", (xxx.shape[0] - np.count_nonzero(yyy))
	# print np.count_nonzero(yyy)

	return xxx, yyy

def print_result(pred, validation_data_y):
    true_positive  = np.count_nonzero(np.logical_and(pred, validation_data_y))
    true_negtive   = np.count_nonzero(np.logical_and((pred==0),(validation_data_y==0)))
    false_positive = np.count_nonzero(np.logical_and((pred==1),(validation_data_y==0)))
    false_negtive  = np.count_nonzero(np.logical_and((pred==0),(validation_data_y==1)))

    error_rate = float(false_negtive + false_positive)/(validation_data_y.shape[0])
    # print "Num of test data:", validation_data_y.shape[0]
    print "true_positive:    %s"%(true_positive)
    print "true_negtive:     %s"%(true_negtive)
    print "false_positive:   %s"%(false_positive)
    print "false_negtive:    %s"%(false_negtive)
    print "total:            %s"%((true_positive)+(true_negtive)+(false_positive)+(false_negtive))

    print "Error Rate        %s"%error_rate
    print "Acuracy:          %s"%(1-error_rate)
    accuracy = 1 - error_rate
    return accuracy

def train_and_print_result(x, y):
	tst_num = int(0.2 * x.shape[0])
	train_data_x       = x[:-tst_num]
	train_data_y       = y[:-tst_num].reshape(-1)
	validation_data_x  = x[-tst_num:]
	validation_data_y  = y[-tst_num:].reshape(-1)	
	start_time = time.time()
	clf= svm.LinearSVC(C=1)
	clf.fit(train_data_x, train_data_y)
	pred = clf.predict(validation_data_x)
	print "--- %s seconds ---" % (time.time() - start_time)
	accuracy = print_result(pred, validation_data_y)
	return accuracy

def train_with_cross_validation(x, y):
	start_time = time.time()
	clf= svm.SVC(cache_size=8000, kernel='linear')
	scores = cross_validation.cross_val_score(clf, x, y, cv=5)
	acuracy = scores.mean()
	print "--- %s seconds ---" % (time.time() - start_time)
	print "Acuracy:          %s"%(acuracy)
	return acuracy

def apply_gabor_filters(x, img_size, filters):
	img = x[0].reshape(img_size, img_size)
	res1 = process(img, filters)
	f = len(res1)
	xx = np.zeros((x.shape[0], f))
	for i in range(x.shape[0]):
	    img = x[i].reshape(img_size, img_size)
	    xx[i] = process(img, filters)
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


def get_pixels_diff(x):
    num = x.shape[1] * (x.shape[1] -1)
    print x.shape[0], num
    num_of_pixels = num *x.shape[0]
    print "num_of_pixels is :", num_of_pixels
    new_x = np.ndarray((x.shape[0], num), dtype='int8')
    cnt = 0
    for i in range (x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[1]-1):
                new_x[i][j*48 + k] = x[i][j] - x[i][k+1]
                cnt = cnt + 1
                if (cnt % 100000) == 0:
                	print "processed %% ", (float(cnt)/num_of_pixels)

face_cascade = cv2.CascadeClassifier('/home/datamining/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

if __name__ == '__main__':
	a_list = []
	# size_list = [20, 24, 28, 32, 36, 40, 44, 48]
	size_list = [48]
	
	(data_x, data_y) = load_smile_data()

	for img_size in size_list:
		x, y = filter_by_frontal_faces(data_x, data_y, img_size)
		
		filters = build_filters(img_size)
		x = apply_gabor_filters(x, img_size, filters)
		# acc=train_and_print_result(x, y)
		y = y.reshape(-1)
		acc = train_with_cross_validation(x, y)
		print "size is: ", img_size, " accuracy is : ", acc

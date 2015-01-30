import cPickle as pickle
import time
import random
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def load_smile_data_from_csv(csv_file):
	print "loading csv file..."
	data = pd.read_csv(csv_file)
	print "loaded."
	for i in range(data.shape[0]):
	    data['emotion'][i] = int(data['emotion'][i])
	    img = data['pixels'][i].split()
	    img = map(int, img)
	    img = np.array(img, dtype='uint8')
	    data['pixels'][i] = img	

	x_1 = data[data.emotion==3]
	x_0 = data[map(lambda x: ((x == 2) or (x == 4) or (x == 5)), data["emotion"])]

	x_1.emotion=1
	x_0.emotion=0

	d = pd.concat([x_0, x_1])
	d = shuffle(d)
	d = shuffle(d)
	d = shuffle(d)
	d = d.reset_index(drop=True)
	x = np.ndarray((d['pixels'].shape[0], d['pixels'][0].shape[0]))
	y = np.ndarray((d['pixels'].shape[0], 1))
	for i in range(d.shape[0]):
	    x[i] = d['pixels'][i]
	    y[i] = d['emotion'][i]
	x = x.astype('uint8')
	y = y.astype('uint8')
	return (x,y)


def save_data_to_pics(x, y):
	f = open("train.txt",'wb+')
	t = open("val.txt", "wb+")
	y = y.astype('uint8')
	y = y.reshape(-1)
	for i in range(x.shape[0]):
	#     print i
	    img = x[i].reshape(48, 48)
	    img = np.array(img)
	    train_num = int(0.8 *x.shape[0])
	#     print train_num
	#     img = img.reshape(48, 48)
	#     if data['Usage'][i]=='Training':
	    if i < train_num:
	        mpimg.imsave("./data/"+str(i)+".png", img, cmap=plt.cm.gray)
	        f.write("./data/"+str(i)+".png " + str(y[i]) + '\n')
	    else:
	        mpimg.imsave("./test/"+str(i)+".png", img, cmap=plt.cm.gray)
	        t.write("./test/"+str(i)+".png " + str(y[i]) + '\n')
	t.close()
	f.close()

def get_frontal_face(img, img_size, scale=1.1, size=5, ):
    face_cascade = cv2.CascadeClassifier('../traditional learning/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.1, 5)
    if len(faces) > 0:
        (x,y,w,h) = faces[0]
        sub_img = img[x:x+w+4, y:y+h+4]
        sub_img = cv2.resize(sub_img, (img_size, img_size))
        return sub_img
    return None

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

def shuffle(df):
    rows = random.sample(df.index, df.shape[0])
    df = df.ix[rows]
    df = df.reset_index(drop=True)
    return df

if __name__ == '__main__':
	original_img_size = 48
	# x, y = load_smile_data_from_pickle("x100.p", "y100.p")
	(data_x, data_y)  = load_smile_data_from_csv('/home/datamining/smile/dataset/fer2013.csv')
	print data_x.shape, data_y.shape
	x, y = filter_by_frontal_faces(data_x, data_y, original_img_size)
	save_data_to_pics(x,y)
import cPickle as pickle
import time
import random
import pandas as pd
from sklearn import svm, metrics
from sklearn.decomposition import PCA
import collections
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_smile_data():
	start_time = time.time()
	x = pickle.load(open( "smile_data.p", "rb"))
	y = pickle.load(open( "smile_labels.p", "rb"))
	x = x.astype("uint8")
	y = y.astype("uint8")
	print "--- Load data in %s seconds ---" % (time.time() - start_time)
	print "data size is : ", x.shape[0]
	print "Positive labels: ", np.count_nonzero(y)
	print "Negtive labels: ", (x.shape[0] - np.count_nonzero(y))
	return (x,y)

def shuffle(df):
    rows = random.sample(df.index, df.shape[0])
    df = df.ix[rows]
    df = df.reset_index(drop=True)
    return df

def save_pickle_data():
	data = pd.read_csv('../fer2013.csv')

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
	d = d.reset_index(drop=True)
	x = np.ndarray((d['pixels'].shape[0], d['pixels'][0].shape[0]))
	y = np.ndarray((d['pixels'].shape[0], 1))
	for i in range(d.shape[0]):
	    x[i] = d['pixels'][i]
	    y[i] = d['emotion'][i]
	x = x.astype('uint8')
	y = y.astype('uint8')
	pickle.dump(x, open("smile_data.p", "wb+"))
	pickle.dump(y, open("smile_labels.p", "wb+")) 
	print y   


if __name__ == '__main__':
	save_pickle_data()

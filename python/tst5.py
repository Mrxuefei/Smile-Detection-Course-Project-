#!/usr/bin/env python

from gabor_classifier import *
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import time
from operator import itemgetter
import os

def get_mouth_pixels(imgs):
	x = np.ndarray((imgs.shape[0],14*24))
	for i in range(imgs.shape[0]):
		img = imgs[i]
		img = img.reshape(48, 48)
		img = img[32:46, 12:36]
		x[i] = img.reshape(-1)
	return x

if __name__ == '__main__':
	
	data_x, data_y = load_smile_data()

	x, y = filter_by_frontal_faces(data_x, data_y, 48)
	x = get_mouth_pixels(x)
	print x.shape
	
	y = y.reshape(-1)
	start_time = time.time()
	clf= svm.SVC()
	scores = cross_validation.cross_val_score(clf, x, y, cv=5)
	acuracy = scores.mean()
	print "--- %s seconds ---" % (time.time() - start_time)
	print "Acuracy: of " + os.path.basename(__file__) + " is :" + str(acuracy)

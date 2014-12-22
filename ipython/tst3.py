#!/usr/bin/env python

from gabor_classifier import *
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from time import time
from operator import itemgetter
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score

def report(grid_scores, n_top=5):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


if __name__ == '__main__':
	a_list = []
	# size_list = [20, 24, 28, 32, 36, 40, 44, 48]
	size_list = [48]
	
	(data_x, data_y) = load_smile_data()
	data_x = data_x.astype('uint8')
	data_y = data_y.astype('uint8')

	for img_size in size_list:
		x, y = filter_by_frontal_faces(data_x, data_y, img_size)

		# x = get_pixels_diff(x)
		
		clf = AdaBoostClassifier(n_estimators=2000)
		scores = cross_val_score(clf, x, y.reshape(-1))
		acc = scores.mean()

		print "tst 3 accuracy is : ", acc
		# filters = build_filters(img_size)
		# x = apply_gabor_filters(x, img_size, filters)
		# acc=train_and_print_result(x, y)
		# y = y.reshape(-1)
		# acc = train_with_cross_validation(x, y)
		# print "size is: ", img_size, " accuracy is : ", acc
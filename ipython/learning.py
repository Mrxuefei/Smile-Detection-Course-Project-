#!/usr/bin/env python

from feature_representation import *
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import time
from operator import itemgetter


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
	(data_x, data_y) = load_smile_data()
	img_size = 48
	x = data_x
	y = data_y
#	x, y = filter_by_frontal_faces(data_x, data_y, img_size)
	filters = build_filters(img_size)
	x = apply_gabor_filters(x, img_size, filters)

	y = y.reshape(-1)
	start_time = time.time()
	clf= svm.LinearSVC(C=10)
	scores = cross_validation.cross_val_score(clf, x, y, cv=5)
	acuracy = scores.mean()
	print "--- %s seconds ---" % (time.time() - start_time)
	print "Acuracy:          %s"%(acuracy)
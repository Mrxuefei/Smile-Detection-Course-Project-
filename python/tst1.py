#!/usr/bin/env python

from gabor_classifier import *
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from time import time
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
	# face_cascade = cv2.CascadeClassifier('/home/datamining/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
	(data_x, data_y) = load_smile_data()
	img_size = 48
	x = data_x
	y = data_y
	x, y = filter_by_frontal_faces(data_x, data_y, img_size)
	filters = build_filters(img_size)
	x = apply_gabor_filters(x, img_size, filters)

	y = y.reshape(-1)
	# start_time = time()

	param_grid = [
	  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
	  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
	 ]

	# run grid search
	svr = svm.SVC()
	grid_search = GridSearchCV(svr, param_grid=param_grid, n_jobs=2)
	start = time()
	grid_search.fit(x, y)

	print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
	      % (time() - start, len(grid_search.grid_scores_)))
	report(grid_search.grid_scores_)	
	# clf= svm.LinearSVC(C=10)
	# scores = cross_validation.cross_val_score(clf, x, y, cv=5)
	# acuracy = scores.mean()
	# print "--- %s seconds ---" % (time.time() - start_time)
	# print "Acuracy:          %s"%(acuracy)
	
	# return acuracy
	# # acc=train_and_print_result(x, y)
	# y = y.reshape(-1)
	# # acc = train_with_cross_validation(x, y)

	# param_grid = [
	#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
	#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
	#  ]

	# # run grid search
	# svr = svm.SVC(cache_size=1000)
	# grid_search = GridSearchCV(svr, param_grid=param_grid, n_jobs=4)
	# start = time()
	# grid_search.fit(x, y)

	# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
	#       % (time() - start, len(grid_search.grid_scores_)))
	# report(grid_search.grid_scores_)

	# print "tst 1  accuracy is : ", acc

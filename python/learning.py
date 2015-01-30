#!/usr/bin/env python

import matplotlib.pyplot as plt

from sklearn import svm, metrics
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier

import time
from operator import itemgetter

from feature_representation import *

def plot_ROC(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(cm):
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def print_result(pred, validation_data_y):
    true_positive  = np.count_nonzero(np.logical_and(pred, validation_data_y))
    true_negtive   = np.count_nonzero(np.logical_and((pred==0),(validation_data_y==0)))
    false_positive = np.count_nonzero(np.logical_and((pred==1),(validation_data_y==0)))
    false_negtive  = np.count_nonzero(np.logical_and((pred==0),(validation_data_y==1)))

    error_rate = float(false_negtive + false_positive)/(validation_data_y.shape[0])
    # print "Num of test data:", validation_data_y.shape[0]
    print "true positive:    %s"%(true_positive)
    print "true negtive:     %s"%(true_negtive)
    print "false positive:   %s"%(false_positive)
    print "false negtive:    %s"%(false_negtive)
    print "total:            %s"%((true_positive)+(true_negtive)+(false_positive)+(false_negtive))

    print "Error Rate        %.1f%%"%(error_rate*100)
    print "Acuracy:          %.1f%%"%(100-error_rate*100)
    accuracy = 1 - error_rate
    return accuracy

def train_and_print_result(x, y, clf, feature_des):
    tst_num = int(0.2 * x.shape[0])
    train_data_x       = x[:-tst_num]
    train_data_y       = y[:-tst_num].reshape(-1)
    validation_data_x  = x[-tst_num:]
    validation_data_y  = y[-tst_num:].reshape(-1)
    print "---Begin to train classifier with %s "%feature_des
    start_time = time.time()
    # clf= svm.LinearSVC(C=1)
    clf.fit(train_data_x, train_data_y)
    pred = clf.predict(validation_data_x)
    y_score = clf.decision_function(validation_data_x)
    fpr, tpr, _ = roc_curve(validation_data_y, y_score)
    roc_auc = auc(fpr, tpr)
    # plot_ROC(fpr, tpr, roc_auc)

    cm = confusion_matrix(validation_data_y, pred)
    accuracy = print_result(pred, validation_data_y)
    print "---Finished train classifier in %.1f seconds ---" % (time.time() - start_time)
    return (accuracy, cm, fpr, tpr, roc_auc)

def train_with_cross_validation(x, y, clf, cv=5):
    print "---Begin to train classifier with cross validation: "
    start_time = time.time()
    # clf= svm.SVC(cache_size=8000, kernel='linear')
    scores = cross_validation.cross_val_score(clf, x, y, cv=5)
    acuracy = scores.mean()
    print "Finished train classifier with cross validation in %.1f seconds ---" % (time.time() - start_time)
    print "Acuracy:          %s"%(acuracy)
    return acuracy


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
	original_img_size = 48
	x, y = load_smile_data_from_pickle("x100.p", "y100.p")

	clf= svm.LinearSVC(C=1000)
	ac, cm, fpr, tpr, roc_auc = train_and_print_result(x, y, clf, "raw pixels")


	filters = build_gabor_filters(original_img_size)
	x2 = apply_gabor_filters(x, original_img_size, filters)
	clf2 = svm.LinearSVC(C=1000)
	ac, cm, fpr, tpr, roc_auc = train_and_print_result(x2, y, clf2, "gabor filters")

	x3 = get_hog_features(x)
	clf3 = svm.LinearSVC(C=1000)
	ac, cm, fpr, tpr, roc_auc = train_and_print_result(x3, y, clf3, "HoG features")

	x4 = get_img_pca(x)
	clf4 = svm.LinearSVC(C=1000)
	ac, cm, fpr, tpr, roc_auc = train_and_print_result(x4, y, clf4, "PCA features")

	x5 = get_img_pca(x)
	clf5 = svm.LinearSVC(C=1000)
	ac, cm, fpr, tpr, roc_auc = train_and_print_result(x5, y, clf5, "area of interest")


	x6 = get_sift_features(x)
	clf6 = svm.LinearSVC(C=1000)
	ac, cm, fpr, tpr, roc_auc = train_and_print_result(x6, y, clf6, "SIFT features")

	param_grid = [
	  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
	  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
	 ]

	# run grid search
	svr = svm.SVC()
	grid_search = GridSearchCV(svr, param_grid=param_grid, n_jobs=2)
	start = time.time()
	grid_search.fit(x2, y.reshape(-1))

	print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
	      % (time.time() - start, len(grid_search.grid_scores_)))
	report(grid_search.grid_scores_)

	x1 = scale_image(x, 24)
	x1 = get_pixels_diff(x1)
	clf1 = AdaBoostClassifier(n_estimators=2000)
	ac, cm, fpr, tpr, roc_auc = train_and_print_result(x1, y, clf1, "pixels differences")
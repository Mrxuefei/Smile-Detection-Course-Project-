#!/usr/bin/env python

from feature_representation import *
from learning import *

original_img_size = 48
x, y = load_smile_data_from_pickle("x100.p", "y100.p")

clf= svm.LinearSVC(C=1000)
ac, cm, fpr, tpr, roc_auc = train_and_print_result(x, y, clf, "raw pixels")

x1 = scale_image(x, 24)
x1 = get_pixels_diff(x1)
clf1 = AdaBoostClassifier(n_estimators=2000)
ac, cm, fpr, tpr, roc_auc = train_and_print_result(x1, y, clf1, "pixels differences")

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
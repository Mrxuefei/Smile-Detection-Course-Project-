from smile import *
data_x, data_y = load_smile_data()
print data_x.shape, data_y.shape
data_x = data_x.astype('uint8')
data_y = data_y.astype('uint8')
print data_x.dtype


from gabor_classifier import *

img_size = 48

x, y = filter_by_frontal_faces(data_x, data_y, img_size)
filters = build_filters(img_size)
x = apply_gabor_filters(x, img_size, filters)

y = y.reshape(-1)
start_time = time.time()
clf= svm.LinearSVC(C=10)
scores = cross_validation.cross_val_score(clf, x, y, cv=5)
acuracy = scores.mean()
print "--- %s seconds ---" % (time.time() - start_time)
print "Acuracy:          %s"%(acuracy)
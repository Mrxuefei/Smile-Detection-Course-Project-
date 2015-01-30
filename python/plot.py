
def plot_100_img(imgs)
	f = plt.figure(figsize=(18, 10))
	for i in range(100):
	    img = imgs[i]
	    f.add_subplot(10, 10, i+1)
	    plt.axis('off')
	    plt.imshow(img, cmap=cm.gray)
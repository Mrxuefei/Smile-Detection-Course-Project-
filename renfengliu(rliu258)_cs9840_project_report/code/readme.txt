
======================For Tradational learning methods======================
1. Before running any coed, the dataset should be downloaded from http://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/download/fer2013.tar.gz and unzip to a folder.

2. In the traditional learning method, following tools and library are used:
    1. Anaconda: https://store.continuum.io/cshop/anaconda/
    2. Scikit-learn Machine learning toolkit: http://scikit-learn.org/ version 0.15.2
    3. OpenCV: http://opencv.org/ version 2.4.9 python module is used.
    4. Pandas :http://pandas.pydata.org/
3. After install those models, modify the code in learning.py in line 101, modify the path for the data source.
4. Type 'python learning.py' in the command line will run the conde. 
5. If need to plot the confusion matrx and ROC curves, uncomment the code under each test.


======================For Deep Learning=====================================
1. The dataset should be downloaded from http://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/download/fer2013.tar.gz and unzip to a folder.
2. Install Caffe deeping learning framework from http://caffe.berkeleyvision.org/
3. Run the python code "python export_csv_to_pic.py" to export the raw data to pictures.
4. Modifiy the script "./prepare_data.sh" and "./train_network.sh" to provide the correct path for the Caffe framework.
5. Run command "./prepare_data.sh" to convert the picture to the data format that the Caffe framework can read.
6. Run command "./train_network.sh" to train the network and see the result.

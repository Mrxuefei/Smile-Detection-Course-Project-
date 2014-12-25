#include "cv.h"
#include "highgui.h"
#include "MyCode.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <windows.h>
#include <string.h>
#include <stdio.h>
#include <io.h>
#include<sstream>
#include <string>  

using namespace cv; 
using namespace std;
/*
int main(int argc, char* argv[]){
	Mat rgba = loadFile("img.xml","img");
	Mat img = cv::imread("1.jpg");
	vector<Mat> chan;
	cv::split(rgba,chan);
	cout<<chan.size()<<endl;
	Mat dest;
	cv::cvtColor(rgba,dest,CV_RGBA2BGR);
	HOGDescriptor* hog = DefaultObjectHOGDescriptor();
	LinearSVM svm;
	svm.load("OBJSVM.xml");
	vector<float> support_vector;
	svm.getSupportVector(support_vector);
	hog->setSVMDetector(support_vector);
	vector<Rect> detectedRect;
	double thresh = 0;
	hog->detectMultiScale(dest,detectedRect,thresh,Size(20,20),Size(),1.2,30);

	cout<<detectedRect.size()<<endl;
	namedWindow( "d", WINDOW_NORMAL);
	imshow("d",dest.t());
	cvWaitKey(0); 
	cv::cvtColor(img,dest,CV_RGB2BGR);
	namedWindow( "d", WINDOW_NORMAL);
	imshow("d",dest.t());
	cvWaitKey(0); 
}*/
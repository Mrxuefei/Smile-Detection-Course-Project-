#include <cv.h>
#include <highgui.h>
#include "MyCode.h"
#include "svm.h"
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"


using namespace std;
using namespace cv;
Mat getInv_feature(Mat& img, int index);

void genFeature(vector<string> paths, Mat& histogram, Mat& inv_features, vector<int>* inv_length)
{
	string line;
	inv_features = Mat(0,0,5);
	inv_features.reserve(2000000000);
	for(int i=0;i<paths.size();i++)
	{
		line = paths.at(i);
		cout<<line<<endl;
		Mat image = cv::imread(line,CV_LOAD_IMAGE_COLOR);
		cv::resize(image,image,Size(200,200));
		Mat local_hist = getHistogram(image, Size(2,2));
		histogram.push_back(local_hist);

		Mat local_inv = getInv_feature(image,2);
		inv_features.push_back(local_inv);
		cout<<inv_features.rows<<"X"<<inv_features.cols<<endl;
		inv_length->push_back(local_inv.rows);
		image.release();
		local_hist.release();
		local_inv.release();
	}
}
Mat getInv_feature(Mat& img, int index)
{
	Mat descriptors;
	switch(index)
	{
		case 1: //surf
			{
				int minHessian = 200;
				SurfFeatureDetector detector(minHessian);
				//MSER
				//FeatureDetector *detect = FeatureDetector.create("SURF");
				std::vector<KeyPoint> keypoints;
				detector.detect(img, keypoints );
				SurfDescriptorExtractor extractor;
				extractor.compute(img, keypoints, descriptors );
				break;
			}
		case 2://sift
			{
				SiftFeatureDetector detector;
				std::vector<KeyPoint> keypoints;
				detector.detect(img,keypoints);
				SiftDescriptorExtractor* extractor = new SiftDescriptorExtractor();
				extractor->compute(img, keypoints, descriptors);
				break;
			}
		default:// hog
			{
				Mat grey;
				cv::cvtColor(img,grey,CV_RGB2GRAY);
				HOGDescriptor* hog = DefaultObjectHOGDescriptor();
				int dec_size = hog->getDescriptorSize();
				vector<float> ders;
				hog->compute(grey,ders,hog->winSize,Size(0,0));
				descriptors = Mat(ders.size()/dec_size,dec_size,CV_32FC1);
				memcpy(descriptors.data,ders.data(),ders.size()*sizeof(float));
			}
	}
	return descriptors;
}
string TextOnImage(string* class_name, Mat result)
{
	string output;
	int n=3;
	for(int i=0;i<n;i++)
	{
		int index = result.at<int>(i);
		output += class_name[index]+"||";
	}
	return output;
}
Mat drawError(vector<int> error,vector<string> paths,string* class_name,Mat probe,double* labels,int n)
{
	int rows = sqrt((double)error.size())+1;
	Size image_size(100,100);
	vector<int>::iterator it = error.begin();
	Mat draw(image_size.width*rows,image_size.height*rows,16,Scalar(0,0,0));
	int index = 0;
	while(it!=error.end())
	{
		Mat image = cv::imread(paths.at(*it));
		//cout<<paths.at(*it)<<endl;	
		Mat probability;
		probe.row(*it).copyTo(probability);
		Mat result;
		sortIdx(probability,result,CV_SORT_EVERY_ROW+CV_SORT_DESCENDING);
		putText(image, TextOnImage(class_name, result),
			Point(500,500), CV_FONT_HERSHEY_COMPLEX, 10, Scalar(0,0,255), 30);
		putText(image, class_name[(int)labels[*it]-1],
			Point(900,900), CV_FONT_HERSHEY_COMPLEX, 10, Scalar(0,0,255), 30);
		cv::resize(image,image,image_size,1);
		int x = index/rows;
		int y = index%rows;
		Rect rect(x*image_size.width,y*image_size.height,image_size.width,image_size.height);
		image.copyTo(draw(rect));
		//imshow("d",draw);
		*it++;
		index++;
	}

	return draw;
}

int main( int argc, wchar_t *argv[ ], wchar_t *envp[ ] )
{
	int k=500;
	string class_name[]={"bread","cake","f c","f p","p1","p2","p3","pp","rad","v s","m s","b san","sau","soup","vegt","waf"};
	vector<double> label_container;
	vector<string> paths;
	TraverseDirectory(L"D:\\classfile",label_container,paths,1);      
	double* labels = &label_container[0];
	Mat histogram, inv_features;
	vector<int>* inv_length = new vector<int>();
	genFeature(paths, histogram, inv_features, inv_length);


	Mat cluster_label,center;
	doKmeans(inv_features,k,cluster_label,center);
	Mat encode_features = encodeAllImages(inv_features, center, inv_length);
	Mat feature;
	cv::hconcat(histogram,encode_features,feature);
	Mat probe;
	vector<int> error = initSVM(feature,labels,class_name,probe);
	Mat draw =  drawError(error,paths,class_name,probe,labels,sizeof(class_name)/sizeof(class_name[0]));
	resize(draw,draw,Size(900,900));
	imshow("d",draw);
	waitKey(0);
	return 0;
}
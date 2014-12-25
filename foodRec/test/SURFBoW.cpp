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

using namespace cv; 
using namespace std; 

Mat getSURFDes(Mat& img)
{
	int minHessian = 400;
	SurfFeatureDetector detector( minHessian );
	FastFeatureDetector detector1;
	ORB detector2;
	StarFeatureDetector detector3;
	BRISK detector4;
	//MSER
	//FeatureDetector *detect = FeatureDetector.create("SURF");
	std::vector<KeyPoint> keypoints;
	Mat drawMat;
	detector.detect( img, keypoints );
	SurfDescriptorExtractor extractor;
	Mat descriptors;
	cv::drawKeypoints(img,keypoints,drawMat,Scalar(255,0,0));
	namedWindow( "d", CV_WINDOW_AUTOSIZE );
	imshow("d",drawMat);
	cvWaitKey(0);  
	extractor.compute(img, keypoints, descriptors );
	return descriptors;
}

void writeIndex(const char* dir)
{
	_finddata_t fileDir;
	string dir2(dir);
	dir2.append("/*.*");
	string file_list = "";
    long lfDir;
	if((lfDir = _findfirst(dir2.c_str(),&fileDir))==-1l)
        printf("No file is found\n");
    else{
//        printf("file list:\n");
        do{
			if(strstr(fileDir.name,"jpg") != NULL)
			{
				string path(dir);
				path.append("/").append(fileDir.name);
				file_list += path+"\n";
				//cout<<path<<endl;
				//printf("%s\n",fileDir.name);
			}
			
        }while( _findnext( lfDir, &fileDir ) == 0 );
    }
    _findclose(lfDir);
	string outpath(dir);
	outpath+="/index.txt";
	ofstream myfile;
	myfile.open(outpath.c_str());
	myfile<<file_list;
	myfile.close();
}
/*
int main(int argc, char* argv[]){
	string str = "C:/Users/Shuang Ao/Desktop/camera";//"D:/xl/index.txt";
	writeIndex(str.c_str());
	str += "/index.txt";
	ifstream out;
	out.open(str.c_str(), ios::in);
    string line;
	Mat surf_des;
	vector<int>* surf_length = new vector<int>();
	while(!out.eof()){
        std::getline(out,line);
        cout <<line<<endl;
		if(line=="")
			break;
		Mat image = cv::imread(line,CV_LOAD_IMAGE_COLOR);
		cv::resize(image,image,Size(800,600));
		cout<< image.rows<<"X"<<image.cols<<endl;
		Mat local_surf_des = getSURFDes(image);
		cout<< local_surf_des.rows<<"X"<<local_surf_des.cols<<endl;
		surf_length->push_back(local_surf_des.rows);
		surf_des.push_back(local_surf_des);
    }
}*/
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

string det_name[] = {	//"FAST",	"STAR",	"SURF",	"ORB",
	"BRISK",//"MSER"
};
int vec[] = {1,1};
string desc_name[] = {//"SIFT" ,
	"SURF"//,"BRIEF" ,"BRISK","ORB","FREAK"
};
double fix_label[] = {1,2,2,2,2,2,2,2,2,2,3,1,3,3,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,2,2};

string str = "D:/xl";//"C:/Users/Shuang Ao/Desktop/camera";//
string index="";
string label_path = "D:/xl/label"+index+".txt";
int k=100;

void writeMatToFile(cv::Mat& m, string filename)
{
	ofstream fout(filename.c_str());

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            fout<<m.at<float>(i,j)<<",";
        }
        fout<<endl;
    }

    fout.close();
}
string double2string(double dbl)
{
	std::ostringstream strs;
	strs << dbl;
	std::string str = strs.str();
	return str;
}
Mat getFeature(Mat& img)
{
	std::vector<KeyPoint> keypoints;
	Mat drawMat;
	for(int i=0;i<vec[0];i++)
	{
		Mat local_drawMat;
		for(int j=0;j<vec[1];j++)
		{
			Mat temp;
			Mat local = img.clone();
			Ptr<FeatureDetector> detector = FeatureDetector::create(det_name[i*3+j]);
			cout<<"start "<<det_name[i*3+j]<<endl;
			detector->detect(local,keypoints);
			cv::drawKeypoints(local,keypoints,temp,Scalar(0,0,255));
			putText(temp, det_name[i*3+j], Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,0,0), 4);
			if(j==0)
				local_drawMat = temp;
			else
				cv::hconcat(local_drawMat,temp,local_drawMat);
		}
		drawMat.push_back(local_drawMat);
	}
	SurfDescriptorExtractor extractor;
	Mat descriptors;
	namedWindow( "d", WINDOW_NORMAL);
	imshow("d",drawMat);
	cvWaitKey(0);  
	extractor.compute(img, keypoints, descriptors );
	return descriptors;
}

Mat getFeatureByIndex(Mat& img,int detector_index, int desc_index)
{
	std::vector<KeyPoint> keypoints;
	Ptr<FeatureDetector> detector = FeatureDetector::create(det_name[detector_index]);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(desc_name[desc_index]);
	detector->detect(img,keypoints);
	Mat descriptors;
	extractor->compute(img, keypoints, descriptors );
	return descriptors;
}

Mat getcolorFeatureByIndex(Mat& img,int detector_index, int desc_index)
{
	vector<Mat> rgb;
	split(img, rgb);
	std::vector<KeyPoint> keypoints_all;
	Ptr<FeatureDetector> detector = FeatureDetector::create(det_name[detector_index]);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(desc_name[desc_index]);
	detector->detect(img,keypoints_all);
	Mat descriptors;
	if(keypoints_all.empty())
		return descriptors;
	for(int i=0;i<3;i++)
	{
		Mat chan_descriptors;
		if(i==0)
			extractor->compute(rgb[i], keypoints_all, descriptors );
		else
		{
			extractor->compute(rgb[i], keypoints_all, chan_descriptors );
			cv::hconcat(descriptors,chan_descriptors,descriptors);
		}
		chan_descriptors.release();
	}
	/*Mat temp;
	extractor->compute(img, keypoints_all, descriptors );
	cv::drawKeypoints(img,keypoints_all,temp,Scalar(0,0,255));
	namedWindow( "d", WINDOW_NORMAL);
	imshow("d",temp);
	cvWaitKey(0);  */
	rgb.clear();
	keypoints_all.clear();
	return descriptors;
}

Mat FeatureGen(string path, int detector_index, int desc_index, vector<int>* f_length)
{
	ifstream out;
	out.open(path.c_str(), ios::in);
	Mat lengthMat;
    string line;
	Mat features;
	int count = 0;
	while(!out.eof()){
        std::getline(out,line);
        cout <<line<<endl;
		if(line=="")
			break;
		Mat image = cv::imread(line,CV_LOAD_IMAGE_COLOR);
		cv::resize(image,image,Size(300,400));
		//cout<< image.rows<<"X"<<image.cols<<endl;
		Mat local_des = getFeatureByIndex(image,detector_index,desc_index);//getcolorFeatureByIndex(image,detector_index,desc_index);//
		cout<<features.rows<<"X"<<features.cols<<endl;
		if(local_des.rows==0)
			cout<< line<<endl;
		f_length->push_back(local_des.rows);
		lengthMat.push_back(local_des.rows);
		features.reserve(features.rows+local_des.rows);
		features.push_back(local_des);
		local_des.release();
		image.release();
		/*count++;
		if(count>100)
			break;
		imshow("d",image);
		cvWaitKey(0);  */
    }
	string tail = "192";
	if(features.cols==64)
		tail = "64";
	writeMatToFile(features,str+"/feature"+tail+".csv");
	writeFile(features,str+"/feature"+tail+".xml","feature");
	writeFile(lengthMat,str+"/length"+tail+".xml","length");
	return features;
}

Mat doSVM(Mat& instances, double* labels, double C=10)
{
	struct svm_parameter param;		// set by parse_command_line
	struct svm_problem prob;		// set by read_problem
	struct svm_model *model;
	struct svm_node *x_space;
	defaultParameter(param);
	param.C = C;
	struct svm_node **_node = new svm_node*[instances.rows];
	for(int i=0;i<instances.rows;i++)
	{
		x_space = new svm_node[instances.cols+1];
		Mat instance = instances.row(i);
		double* valueArray = new double[instances.cols];
		valueArray = instances.ptr<double>(i);
		for(int j=0;j<instances.cols;j++)
		{
			x_space[j].index=j;
			x_space[j].value = valueArray[j];
		}
		x_space[instances.cols].index=-1;
	    x_space[instances.cols].value = -1;
		_node[i] = x_space;
	}
	prob.l = instances.rows;
	prob.y = labels;
	prob.x = _node;
	double* target = new double[prob.l];
	//CvSVMParams cvParams =  AutoSVMParameter(instances,  labels);
	//copyPara(param, cvParams);

	svm_cross_validation(&prob,&param,5,target);
	int total_correct = 0;
	for(int i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
	model = svm_train(&prob,&param);
	svm_save_model("model.txt",model);
	Mat confusionMatrix(model->nr_class,model->nr_class,CV_32F,Scalar(0));
	cout<<"optimized C: "<<param.C<<endl;
	printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	for(int i=0;i<prob.l;i++)
		confusionMatrix.at<float>((int)prob.y[i]-1,(int)target[i]-1) +=1;
	cout<<confusionMatrix<<endl;
	return confusionMatrix;
}

void test()
{
	//writeIndex(str.c_str());
	//str += "/index.txt";
	FileStorage fs(str+"/cf.xml", FileStorage::WRITE);
	for(int detector_index = 0;detector_index < sizeof(det_name)/sizeof(det_name[0]);detector_index++)
	{
		for(int desc_index = 0;desc_index < sizeof(desc_name)/sizeof(desc_name[0]);desc_index++)
		{
			vector<int>* f_length = new vector<int>();
			cout<<det_name[detector_index]+"+"+desc_name[desc_index]<<endl;
			
			Mat features = FeatureGen( str+"/index"+index+".txt",  detector_index,  desc_index,  f_length);
			double* labels = fix_label;//new double[f_length->size()];//
			//initLabel(label_path, labels);
			Mat center;
			features.convertTo(features,CV_32F);
			doKmeans(features,k,Mat(),center);
			writeMatToFile(center,str+"/center.csv");
			Mat feature_code = encodeAllImages(features, center, f_length);
			Mat fc_mat;
			fc_mat.push_back(doSVM(feature_code, labels));
			fs << det_name[detector_index]+"_"+desc_name[desc_index] << fc_mat;
			center.release();
			features.release();
			fc_mat.release();
		}
	}
	fs.release();
}

void statistics()
{
	FileStorage fs(str+"/cf.xml", FileStorage::READ);
	stringstream result;
	for(int detector_index = 0;detector_index < sizeof(det_name)/sizeof(det_name[0]);detector_index++)
	{
		
		for(int desc_index = 0;desc_index < sizeof(desc_name)/sizeof(desc_name[0]);desc_index++)
		{
			Mat data;
			fs[det_name[detector_index]+"_"+desc_name[desc_index]] >> data;
			//cout<<sum(data.diag())[0]/sum(data)[0]<<endl;
			result<<(sum(data.diag())[0]/sum(data)[0])<<" ";
		}
		result<<endl;
	}
	ofstream myfile;
	myfile.open(str+"/sta.txt");
	myfile<<result.str();
	myfile.close();
}

void show()
{
	ifstream out;
	out.open(str+"/index.txt", ios::in);
    string line;
	Mat features;
	while(!out.eof()){
        std::getline(out,line);
        cout <<line<<endl;
		if(line=="")
			break;
		Mat image = cv::imread(line,CV_LOAD_IMAGE_COLOR);
		cv::resize(image,image,Size(600,800));
		cout<< image.rows<<"X"<<image.cols<<endl;
		Mat local_des = getFeature(image);
		cout<< local_des.rows<<"X"<<local_des.cols<<endl;
    }
}
string cols = "64";
void load()
{
	Mat features = loadFile(str+"/feature"+cols+".xml","feature");
	Mat lengthMat= loadFile(str+"/length.xml","length");
	vector<int>* f_length = new vector<int>();
	for(int i=0;i<lengthMat.rows;i++)
		f_length->push_back(lengthMat.at<int>(i,0));
	cout<<f_length->size()<<endl;
	double* labels = new double[f_length->size()];//fix_label;//
	initLabel(label_path, labels);
	Mat center;
	features.convertTo(features,CV_32F);
	doKmeans(features,k,Mat(),center);
	writeMatToFile(center,str+"/center.csv");
	Mat feature_code = encodeAllImages(features, center, f_length);
	Mat fc_mat;
	fc_mat.push_back(doSVM(feature_code, labels));
	writeFile(fc_mat,str+"/cf"+cols+".xml","cf");
	center.release();
	features.release();
	fc_mat.release();

}

void cmp()
{
	string tag[] = {"64","192"};
	double C[] = {10,20,30,40};
	FileStorage fs(str+"/cmp.xml", FileStorage::WRITE);
	double* result = new double[sizeof(tag)/sizeof(tag[0])*sizeof(C)/sizeof(C[0])];
	int count = 0;
	for(int i=0;i< sizeof(tag)/sizeof(tag[0]);i++)
	{
		Mat features = loadFile(str+"/feature"+tag[i]+".xml","feature");
		Mat lengthMat= loadFile(str+"/length.xml","length");
		vector<int>* f_length = new vector<int>();
		for(int i=0;i<lengthMat.rows;i++)
			f_length->push_back(lengthMat.at<int>(i,0));
		cout<<f_length->size()<<endl;
		double* labels = new double[f_length->size()];//fix_label;//
		initLabel(label_path, labels);
		Mat center;
		features.convertTo(features,CV_32F);
		doKmeans(features,k,Mat(),center);
		Mat feature_code = encodeAllImages(features, center, f_length);
		for(int j=0;j<sizeof(C)/sizeof(C[0]);j++)
		{
			Mat cf_mat = doSVM(feature_code, labels,C[j]);
			double acc = (sum(cf_mat.diag())[0]/sum(cf_mat)[0]);
			result[count] = acc;
			string p = "rs_"+tag[i]+"_"+double2string(C[j]);
			fs<<p<<cf_mat;
			count++;
		}
	}
	Mat acc_mat = Mat(sizeof(tag)/sizeof(tag[0]),sizeof(C)/sizeof(C[0]),CV_64F,result);
	fs<<"acc_mat_cmp"<<acc_mat;
	fs.release();
}
/*
void t()
{
	double* p = new double[100];
	for(int i=0;i<100;i++)
		p[i] = i;
	Mat acc_mat = Mat(10,10,CV_64F,p);
	FileStorage fs(str+"/t.xml", FileStorage::WRITE);
	fs<<"X"<<acc_mat;
	fs.release();
}*/
/*
int main(int argc, char* argv[]){
	//test();
	//load();
	//statistics();
	//show();
	cmp();
	//t();
	getchar();
}*/
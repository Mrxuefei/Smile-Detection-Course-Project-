#include <cv.h>
#include <highgui.h>
#include "MyCode.h"
#include "svm.h"
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <numeric>
 
using namespace cv; 
using namespace std; 

//string class_name[]={"banana","burger","fries","pasta","rice","salad","tomato"};
string class_name[]={"banana","burger","fries","pasta","rice","salad","tomato"};

extern cv::Mat svm_prob_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);
void defaultParameter(svm_parameter& param)
{
	param.svm_type = C_SVC;
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 50;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
}
void copyPara(svm_parameter& param, CvSVMParams cvParams)
{
	param.degree = (int)cvParams.degree;
	param.gamma = cvParams.gamma;	// 1/num_features
	param.coef0 = cvParams.coef0;
	param.nu = cvParams.nu;
	param.C = cvParams.C;
	param.p = cvParams.p;
}
CvSVMParams AutoSVMParameter(Mat& instances, double* labels)
{
	CvSVMParams params;
	float* f_labels = new float[instances.rows];
	for(int i=0;i<instances.rows;i++)
		f_labels[i] = (float)labels[i];
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 1e-6);
	Mat labelMat(instances.rows,1,CV_32F,f_labels);
	Mat feature;
	instances.convertTo(feature,CV_32F);
	CvSVM svm;
	svm.train_auto(feature, labelMat, Mat(), Mat(), params,5);  
	params = svm.get_params();
	svm.save("opencv_svm.xml");
	return params;
}
vector<int> initSVM(Mat& instances, double* labels,string* class_name)
{
	struct svm_parameter param;		// set by parse_command_line
	struct svm_problem prob;		// set by read_problem
	struct svm_model *model;
	struct svm_node *x_space;
	defaultParameter(param);
	struct svm_node **_node = new svm_node*[instances.rows];
	vector<int> error_list;
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
			else
				error_list.push_back(i);
	model = svm_train(&prob,&param);
	svm_save_model("model.txt",model);
	Mat confusionMatrix(model->nr_class,model->nr_class,CV_32F,Scalar(0));
	cout<<"optimized C: "<<param.C<<endl;
	printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	for(int i=0;i<prob.l;i++)
		confusionMatrix.at<float>((int)prob.y[i]-1,(int)target[i]-1) +=1;
	for(int i=0;i<model->nr_class;i++)
		cout<<class_name[i]<<":"<<confusionMatrix.row(i)<<endl;
	return error_list;
}
vector<int> initSVM(Mat& instances, double* labels,string* class_name,Mat& probe)
{
	struct svm_parameter param;		// set by parse_command_line
	struct svm_problem prob;		// set by read_problem
	struct svm_model *model;
	struct svm_node *x_space;
	defaultParameter(param);
	struct svm_node **_node = new svm_node*[instances.rows];
	vector<int> error_list;
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

	probe = svm_prob_cross_validation(&prob,&param,5,target);
	cout<<probe<<endl;
	//svm_cross_validation(&prob,&param,5,target);
	int total_correct = 0;
	for(int i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
			else
				error_list.push_back(i);
	model = svm_train(&prob,&param);
	svm_save_model("model.txt",model);
	Mat confusionMatrix(model->nr_class,model->nr_class,CV_32F,Scalar(0));
	cout<<"optimized C: "<<param.C<<endl;
	printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	for(int i=0;i<prob.l;i++)
		confusionMatrix.at<float>((int)prob.y[i]-1,(int)target[i]-1) +=1;
	for(int i=0;i<model->nr_class;i++)
		cout<<class_name[i]<<":"<<confusionMatrix.row(i)<<endl;
	writeFile(confusionMatrix,"cf_mat.xml","cf_mat");
	return error_list;
}




void initData(String path, Mat& histogram, Mat& hogs, vector<int>* hog_length)
{
	ifstream out;
	out.open(path.c_str(), ios::in);
    string line;
	vector<string> class_name;
    while(!out.eof()){
        std::getline(out,line);
        cout <<line<<endl;
		cout << histogram.rows<<endl;
		Mat image = cv::imread(line,CV_LOAD_IMAGE_COLOR);
		cv::resize(image,image,Size(200,200));
		Mat local_hist = getHistogram(image, Size(2,2));
		//cout<< local_hist<<endl;
		histogram.push_back(local_hist);
		
		Mat local_hogs = getHogs(image);
		hog_length->push_back(local_hogs.rows);
		hogs.push_back(local_hogs);
		/*
		Mat local_surf = getSURFDes(image);
		hogs.push_back(local_surf);
		hog_length->push_back(local_surf.rows);*/
    }
    out.close();
}

void initLabel(String path, double* label)
{
	ifstream out;
	out.open(path.c_str(), ios::in);
    string line;
	int i=0;
	 while(!out.eof()){
		 std::getline(out,line);
		 label[i] = atof(line.c_str());
		 i++;
	 }
}

void writeFile(Mat data, string path,string tag)
{
	FileStorage fs(path, FileStorage::WRITE);
	fs << tag << data;
	fs.release();
}
Mat loadFile(string path, string tag)
{
	FileStorage fs(path, FileStorage::READ);

	Mat data;
	fs[tag] >> data;
	return data;
}


/*
int main(int argc, char* argv[])
{
	
	ifstream out;
    string str = "D:/xl/index.txt";
	string label_path = "D:/xl/label.txt";
	
	Mat histogram;
	Mat hogs;
	vector<int> *hog_length = new vector<int>();
	initData( str,  histogram, hogs, hog_length);
	double* labels = new double[histogram.rows];
	initLabel(label_path,labels);
	cout<<std::accumulate(hog_length->begin(),hog_length->end(),0)<<endl;

	Mat cluster_label,center;
	int k=200;
	doKmeans(hogs,k,cluster_label,center);
	writeFile(center,"test.xml","center");
	writeFile(hogs, "hog.xml","hog");
	writeFile(histogram, "histogram.xml","histogram");
	Mat hogf = encodeAllImages(hogs, center, hog_length);
	//cout<<hogf.row(1)<<endl;


	Mat feature;
	cv::hconcat(histogram,hogf,feature);
	//cout<<feature.row(1)<<endl;
	initSVM(feature,labels,class_name);

	getchar();
    return 0;
}*/
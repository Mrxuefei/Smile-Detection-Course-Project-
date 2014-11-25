#include "cv.h"
#include "highgui.h"
#include "MyCode.h"
#include <sys/types.h> 
#include <opencv2/ml/ml.hpp>
#include <windows.h>
#include <string.h>
#include <stdio.h>
#include <io.h>

using namespace cv; 
using namespace std; 

void LinearSVM::getSupportVector(std::vector<float>& support_vector) const {

    int sv_count = get_support_vector_count();
    const CvSVMDecisionFunc* df = decision_func;
    const double* alphas = df[0].alpha;
    double rho = df[0].rho;
    int var_count = get_var_count();
    support_vector.resize(var_count, 0);
    for (unsigned int r = 0; r < (unsigned)sv_count; r++) {
      float myalpha = alphas[r];
      const float* v = get_support_vector(r);
      for (int j = 0; j < var_count; j++,v++) {
        support_vector[j] += (-myalpha) * (*v);
      }
    }
    support_vector.push_back(rho);
}

void findFiles(const char* dir,vector<string>& paths)
{
	_finddata_t fileDir;
    long lfDir;
    if((lfDir = _findfirst(dir,&fileDir))==-1l)
        printf("No file is found\n");
    else{
//        printf("file list:\n");
        do{
			if(strstr(fileDir.name,"jpg") != NULL)
			{
				string path(fileDir.name);
				paths.push_back(path);
				//printf("%s\n",fileDir.name);
			}
			
        }while( _findnext( lfDir, &fileDir ) == 0 );
    }
    _findclose(lfDir);
}
HOGDescriptor* DefaultObjectHOGDescriptor()
{
	//cv::Size winSize(128,128);
	cv::Size winSize(128,128);
	cv::Size blockSize(winSize.width/2, winSize.height/2);
	cv::Size cellSize(blockSize.width/2, blockSize.height/2);
	cv::Size blockStride(blockSize.width,blockSize.height);
	int nbins = 9;
	HOGDescriptor* hog = new HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins);
	return hog; 
}
HOGDescriptor* MyObjectHOGDescriptor(Size win)
{
	//cv::Size winSize(128,128);
	cv::Size winSize(win);
	cv::Size blockSize(winSize.width/2, winSize.height/2);
	cv::Size cellSize(blockSize.width/2, blockSize.height/2);
	cv::Size blockStride(blockSize.width,blockSize.height);
	int nbins = 9;
	HOGDescriptor* hog = new HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins);
	return hog;
}

Mat ObjtHogs(Mat& origImg)
 {
	Mat grey;
	cv::cvtColor(origImg,grey,CV_RGB2GRAY);
	HOGDescriptor* hog = DefaultObjectHOGDescriptor();
	vector<float> ders;
	vector<Point>locs;
	int dec_size = hog->getDescriptorSize();
	//HOGDescriptor* hog;
	hog->compute(grey,ders,hog->winSize,Size(0,0),locs);
	Mat trans(ders.size()/dec_size,dec_size,CV_32FC1);
	memcpy(trans.data,ders.data(),ders.size()*sizeof(float));
	return trans;
 }

void trainSVM(Mat feature,Mat labels, CvSVM& svm)
{
	CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 1e-6);
	svm.train_auto(feature, labels, Mat(), Mat(), params,10);  
	params = svm.get_params();
	svm.train(feature, labels, Mat(), Mat(), params);
	svm.save("OBJSVM.xml");
}


void Pyramid_detector(Mat& img,LinearSVM& svm)
{
	
	HOGDescriptor* hog = MyObjectHOGDescriptor(Size(128,128));
	vector<float> support_vector;
	svm.getSupportVector(support_vector);
	hog->setSVMDetector(support_vector);
	vector<Rect> detectedRect;
	double thresh = 0;
	int start = cv::getTickCount();
	hog->detectMultiScale(img,detectedRect,thresh,Size(10,10),Size(),1.2,15);
	int finish = cvGetTickCount();
	printf("detection time = %.3f\n", (float)(finish - start) / (float)(cvGetTickFrequency() * 1000000.0));
	cout<<detectedRect.size()<<" locations"<<endl;
	for(int i=0;i<detectedRect.size();i++){
		cv::rectangle(img,detectedRect.at(i),Scalar(0,255,0),3);
		//cout<<detectedRect.at(i).width<<"X"<<detectedRect.at(i).height<<endl;
	}
	
	/*start = cv::getTickCount();
	hog->detectMultiScale(img,detectedRect,thresh,Size(20,20),Size(),1.1,0);
	finish = cvGetTickCount();
	cout<<detectedRect.size()<<" locations"<<endl;
	for(int i=0;i<detectedRect.size();i++){
		cv::rectangle(img,detectedRect.at(i),Scalar(0,255,0),3);
		cout<<detectedRect.at(i).width<<"X"<<detectedRect.at(i).height<<endl;
	}*/
	printf("detection time = %.3f\n", (float)(finish - start) / (float)(cvGetTickFrequency() * 1000000.0));
	if(detectedRect.size()>0)
	{
		imshow("d",img);
		cvWaitKey(0);  
	}
}
void detectPath(String path, LinearSVM& svm)
{
	Mat img = imread(path);
	if(img.rows<DefaultObjectHOGDescriptor()->winSize.width||img.cols<DefaultObjectHOGDescriptor()->winSize.height)
		cv::resize(img,img,Size(DefaultObjectHOGDescriptor()->winSize.width,DefaultObjectHOGDescriptor()->winSize.height));
	//cv::resize(img,img,Size(800,600));
	Pyramid_detector(img,svm);
}
/*
int main(int argc, char* argv[]){
	string path = "D:\\deeplearningcode\\data\\food100\\objdetect\\";
	
	string search_path = path;
	const char* dir=search_path.append("*.*").c_str();
	
	vector<string> paths;
	findFiles(dir,paths);
	Mat feature;
	float* labels = new float[paths.size()];
	for(int i=0;i<paths.size();i++)
	{
		cout<<i<<": "<<path+paths.at(i)<<endl;
		Mat img = cv::imread(path+paths.at(i));
		feature.push_back(ObjtHogs(img));
		size_t _find = paths.at(i).find("pos");
		if(_find==0)
			labels[i] = 1;
		else
			labels[i]= -1;
		cout<<labels[i]<<endl;
	}
	Mat labelMat(paths.size(),1,CV_32FC1,labels);
	LinearSVM svm;
	trainSVM(feature,labelMat,svm);
	cout<<"done"<<endl;
	
	LinearSVM svm;
	svm.load("OBJSVM.xml");
	string dd = "D:\\deeplearningcode\\data\\food100\\1\\";
	//string dd = "D:\\xl\\neg-obj\\";
	string dd2 = dd;
	const char* dest = dd2.append("*.*").c_str();
	vector<string> d_paths;
	findFiles(dest,d_paths);
	for(int i=0;i<d_paths.size();i++){
		cout<<dd+d_paths.at(i)<<endl;
		detectPath(dd+d_paths.at(i), svm);
	}
	getchar();
    return 0;
}*/
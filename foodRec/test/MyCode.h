#include "cv.h"
#include "svm.h"
#include "highgui.h"
#include <opencv2/ml/ml.hpp>
#include "windows.h"
#include <string.h>
#include <Strsafe.h>
using namespace cv; 
using namespace std;

Mat getHistogram( Mat& origImg, Size sz);

Mat getHogs(Mat& origImg);
Mat encodeImage(Mat& feature, Mat& center);
void doKmeans(Mat instances, int k, Mat& labels, Mat& centers);
Mat encodeAllImages(Mat& features, Mat center, vector<int>* length);
Mat loadFile(string path, string tag);

Mat getPatchHistogram(Mat& origImg);
Mat getSURFDes(Mat& img);

void findFiles(const char* dir,vector<string>& paths); 
void writeIndex(const char* dir);
vector<int> initSVM(Mat& instances, double* labels, string* class_name);
vector<int> initSVM(Mat& instances, double* labels,string* class_name,Mat& probe);
void defaultParameter(svm_parameter& param);


void writeFile(Mat data, string path,string tag);
void initLabel(String path, double* label);
void findFiles(const char* dir,vector<string>& paths);
int TraverseDirectory(wchar_t Dir[MAX_PATH], vector<double>& label,vector<string>& paths,double i=1);    

HOGDescriptor* DefaultObjectHOGDescriptor();
class LinearSVM: public CvSVM {
public:
  void getSupportVector(std::vector<float>& support_vector) const;
};
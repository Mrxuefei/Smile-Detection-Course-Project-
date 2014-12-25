#include "cv.h"
#include "highgui.h"
#include "MyCode.h"
#include "svm.h"
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <numeric>
 
using namespace cv; 
using namespace std; 
/*
int main(int argc, char* argv[]){
	svm_model* model = svm_load_model("d:/model.txt");
	FileStorage fs("lib2opencv.xml", FileStorage::WRITE);
	fs.writeObj("lib2opencv.xml",model->SV);
	fs.release();
	getchar();
}*/
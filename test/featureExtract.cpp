#include <cv.h>
#include <highgui.h>

using namespace cv; 
using namespace std; 

 Mat getPatchHistogram(Mat& origImg)
{
	int bins = 4;
	int nbins[] = {bins,bins,bins};
	const int channels[] = {0,1,2};
	float rrange[] = {0,256};
	float grange[] = {0,256};
	float brange[] = {0,256};
	const float *ranges[] = {rrange,grange,brange};
	Mat histMat;
	calcHist(&origImg,1,channels,Mat(),histMat,3,nbins,ranges,true,false);
	/*float* data = new float[histMat.total()];
	int index = 0;
	int count = 0;
	
	for(int i=0;i<bins;i++)
	{
		for(int j=0;j<bins;j++)
		{
			for(int k=0;k<bins;k++)
			{
				cout<<histMat.at<float>(i,j,k)<<", ";
				data[count] = histMat.at<float>(i,j,k);
				index +=histMat.at<float>(i,j,k);
				count++;
			}
			cout<<endl;
		}
		cout<<endl;
		cout<<endl;
	}*/
	return histMat;
}
Mat getNbins(Mat& subMat, int nbins)
	{
		int n=2120;
		double* hist = new double[nbins*nbins*nbins];
		for(int i = 0; i < nbins*nbins*nbins; i++) 
			hist[i]=0;
		subMat.convertTo(subMat,CV_32FC3);
		subMat = subMat.reshape(1, subMat.rows*subMat.cols); //[M*N]*3 
		//cout<<subMat.row(n)<<endl;
		double interval = 256/nbins;
		subMat = subMat/interval;
		//cout<<subMat.row(n)<<endl;
		for (int i = 0; i < subMat.rows; i++) { //stored in BGR
			int bIndex = (int)subMat.at<float>(i, 0);
			int gIndex = (int)subMat.at<float>(i, 1);;
			int rIndex = (int)subMat.at<float>(i, 2);;
			hist[bIndex*nbins*nbins+gIndex*nbins+rIndex]++;
		}
		Mat hist2(1,64,CV_64F,hist);
		return hist2;
	}

Mat getHistogram(Mat& origImg,Size sz)
{
	int r = origImg.rows/sz.height;
	int c = origImg.cols/sz.width;
	Mat hist;
	//cout<<origImg.channels();
	for(int i=0;i<sz.height;i++)
	{
		for (int j=0;j<sz.width;j++)
		{
			Rect rect(c*j,r*i,c,r);
			Mat sub_img = origImg(rect);
			//Mat sub_hist = getNbins(sub_img,4);
			Mat sub_hist = getPatchHistogram(sub_img);
			float* hist2 = (float*)sub_hist.data;
			Mat local_hist(1,64,CV_32F,hist2);
			//cout<<"sub_hist"<<sub_hist<<endl;
			//hist.push_back(sub_hist);
			hist.push_back(local_hist);
		}
	}
	hist = hist.reshape(1,1);
	//cout<<sum(hist)<<endl;
	hist.convertTo(hist,CV_64FC1);
	hist = hist/sum(hist)[0];
	return hist;
}

 Mat getHogs(Mat& origImg)
 {
	cv::Size winSize(128,128);
	if(origImg.rows<winSize.height||origImg.cols<winSize.width)
		cv::resize(origImg,origImg,Size(200,200));
	Mat grey;
	cv::cvtColor(origImg,grey,CV_RGB2GRAY);
	cv::Size blockSize(winSize.width/2, winSize.height/2);
	cv::Size cellSize(blockSize.width/2, blockSize.height/2);
	cv::Size blockStride(blockSize.width,blockSize.height);
	int nbins = 8;
	HOGDescriptor* hog = new HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins);
	vector<float> ders;
	vector<Point>locs;
	int dec_size = hog->getDescriptorSize();
	//HOGDescriptor* hog;
	hog->compute(grey,ders,winSize,Size(0,0),locs);
	Mat trans(ders.size()/dec_size,dec_size,CV_32FC1);
	memcpy(trans.data,ders.data(),ders.size()*sizeof(float));
	return trans;
 }

 void doKmeans(Mat instances, int k, Mat& labels, Mat& centers)
 {
	 cv::kmeans(instances,k,labels,TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
		3, KMEANS_PP_CENTERS, centers);  //聚类3次，取结果最好的那次，聚类的初始化采用PP特定的随机算法。
 }

 Mat encodeImage(Mat& feature, Mat& center)
 {
	 Mat result;
	 double* dist = new double[center.rows];
	 for(int i=0;i<center.rows;i++)
	 {
		 Mat long_center ;
	//	 cv::repeat(center.row(i),feature.rows,1,long_center);
		 double min = DBL_MAX;
		 for(int j=0;j<feature.rows;j++)
		 {
			 double temp = cv::norm(feature.row(j),center.row(i));
			 if(temp<min)
				 min = temp;
		 }
		 dist[i]=min;
	 }
	 result = Mat(1,center.rows ,CV_64FC1,dist);
	 //cout<<"result"<<result<<endl;
	 return result;
 }
 Mat encodeAllImages(Mat& features, Mat center, vector<int>* length )
 {
	 Mat result(length->size() ,center.rows,CV_64FC1);
	 int start = 0;
	 for (int i=0; i< result.rows;i++)
	 {
		 int end = start + length->at(i);
		 cout<<"from "<<start<<" to "<<end<<endl;
		 Range r(start,end);
		 Mat img_feature;
		 if(start == end)
		 {
			 img_feature = Mat::zeros(Size(center.rows,1 ),CV_64FC1);
			 img_feature.copyTo(result.row(i));
		 }
		 else
		 {
			img_feature = Mat(features, r);
			encodeImage(img_feature,center).copyTo(result.row(i));
		 }
		 //cout<<"All Img:"<<result.row(i)<<endl;
		 start = end;
	 }
	 return result;
 }


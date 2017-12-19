// This is the LBP program for MORPH Album 2 Database (small age gap).
// The dataset "MORPH_PROC" contains 700 images of 100 subjects.
// Each subject has 6 training images at younger ages and 1 testing image at older age. 
// The age gaps between training & testing images are 1~5 years.
// Author: Minghao Ye

#include "opencv2\opencv.hpp"
#include <vector>
#include <io.h>

using namespace std;
using namespace cv;

#define DEST_WIDTH 140 // Image width
#define DEST_HEIGHT 140 // Image height
#define PEOPLENUM 100 // Number of subject
#define ALLPICNUM 7 // Number of images per subject
#define TRAINNUM  6 // Number of training samples per subject

const int xblocknum = 4;// X direction division
const int yblocknum = 5;// Y direction division

// Weights setting for the blocks. 
// These are the best weights settings obtained from experiments. 

// 10x10 division: 57% Accuracy
/*const float weight[10][10] = { { 11,12,9,4,3,11,5,6,3,4 },
{ 0,0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0,0 } };*/

// 9x9 division: 59% Accuracy
/*const float weight[9][9] = { { 5,10,19,12,8,4,12,11,5 },
{ 0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,0 } };*/

// 8x8 division: 63% Accuracy
/*const float weight[8][8] = { { 10,13,14,9,8,11,13,7 },
{ 0,0,0,0,0,0,0,1 },
{ 0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0 },
{ 0,0,0,0,0,0,0,0,} };*/

// 7x7 division: 60% Accuracy
/*const float weight[7][7] = {{10,24,12,11,8,18,4},
{0,0,0,0,0,0,0},
{0,0,0,0,0,0,0},
{0,0,0,0,0,0,0},
{0,0,0,0,0,0,0},
{0,0,0,0,0,0,0},
{0,0,0,0,0,0,0}};*/

// 6x6 division: 64% Accuracy
/*const float weight[6][6] = { { 22,21,8,10,15,9 },
{ 0,0,0,0,0,0},
{ 0,0,0,0,0,0 },
{ 0,0,0,0,0,0 },
{ 0,0,0,0,0,0 },
{ 0,0,0,0,0,0 } }; */

// 5x6 division: 62% Accuracy
/*const float weight[5][6] = { { 27,24,18,13,13,10 },
{ 0,0,0,0,0,0 },
{ 0,0,0,0,0,0 },
{ 0,0,0,0,0,0 },
{ 0,0,0,0,0,0 } };*/

// 6x5 division: 61% Accuracy
/*const float weight[6][5] = { { 28,22,9,13,9 },
{ 0,0,0,0,0 },
{ 0,0,0,0,0 },
{ 0,0,0,0,0 },
{ 0,0,0,0,0 },
{ 0,0,0,0,0 } };*/

// 5x5 division: 66% Accuracy
/*const float weight[5][5] = { { 25,25,11,14,13 },
{ 0,0,0,0,0 },
{ 0,0,0,0,0 },
{ 0,0,0,0,0 },
{ 0,0,0,0,0 } };*/

// 5x4 division: 59% Accuracy
/*const float weight[5][4] = { { 21,20,18,15 },
{ 0,0,0,0 },
{ 0,0,0,0 },
{ 0,0,0,0 },
{ 0,0,0,0 } };*/ 

// 4x5 division: 70% Accuracy (max)
const float weight[4][5] = { { 24,22,12,19,15 },
{ 0,0,0,0,0 },
{ 0,0,0,0,0 },
{ 0,0,0,0,0 } };

// 4x4 division: 63% Accuracy
/*const float weight[4][4] = {{25,21,20,18},
{0,0,0,0},
{0,0,0,0},
{0,0,0,0}};*/

// 3x3 division: 55% Accuracy
/*const float weight[3][3] = {{29,21,16},
{0,0,0},
{0,0,0}};*/ 

// 2x2 division: 43% Accuracy
/*const float weight[2][2] = { { 25,25 },
{ 0,0} };*/ 

// 1x1 division: 14% Accuracy
/*const float weight[1][1] = {{1}};*/

// 58 uniform patterns
// Each uniform pattern occupies 1 bin
const int LBP_NUM[]={0,1,2,3,4,6,7,8,12,14,15,16,24,28,30,31,32,48,56,60,62,63,64,96,112,120,124,126,127,128,129,131,135,
143,159,191,192,193,195,199,207,223,224,225,227,231,239,240,241,243,247,248,249,251,252,253,254,255};
// Other non-uniform patterns occupy 1 bin£¬totally 59 bins in the LBP histogram.

uchar maptable[256] = {0};

vector<Mat> trainfeaturelist;// Feature list of training sample
vector<int> trainlabels;


// Copy IplImage
int imageClone(IplImage* pi,IplImage** ppo)  
{
	if (*ppo) {
		cvReleaseImage(ppo);                //  Release original image
	}
	(*ppo) = cvCloneImage(pi);              //  Copy new image
	return(1);
}

//  Replace IplImage
int  imageReplace(IplImage* pi,IplImage** ppo)  
{
	if (*ppo) 
		cvReleaseImage(ppo);                //  Release original image
	(*ppo) = pi;                            //  Change name
	return(1);
}


// This function can obtain all picture files from the folder. Such as .bmp, .jpg, .png
void getPicFiles(string path, vector<string>& files,vector<string>& filenames)
{
	long   hFile   =   0;
	// file information
	struct _finddata_t fileinfo;
	string p;
	if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)
	{
		do
		{
			// If it is a directory, iterate
			// If not, add to the list
			if((fileinfo.attrib &  _A_SUBDIR))
			{
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
					getPicFiles(path+"\\"+fileinfo.name, files,filenames);
			}
			else
			{

				char *pp;
				pp = strrchr(fileinfo.name,'.');// Locate the last occurrence position
				if (_stricmp(pp,".bmp")==0 || _stricmp(pp,".jpg")==0 || _stricmp(pp,".png")==0 )// If it is picture files, process it
				{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));// Save filename with datapath
					filenames.push_back(fileinfo.name);// Only save filename
				}
			}
		}while(_findnext(hFile, &fileinfo)  == 0);
		_findclose(hFile);
	}
}


// This function can preprocess images, such as rgb2gray, normalization, etc.
void preproc(IplImage* inimage,IplImage** outimage,bool bshow=false)
{
	IplImage* gray = cvCreateImage( cvSize(inimage->width,inimage->height), 8, 1 );// Gray-scale image
	if(inimage->nChannels == 3)
	{
		cvCvtColor( inimage, gray, CV_BGR2GRAY );// Transform original RGB image to gray-scale image
	}
	else
	{
		cvCopy(inimage,gray);
	}

	if(bshow)
	{
		cvNamedWindow("Gray-scale image",0);
		cvShowImage("Gray-scale image",gray);
	}

	IplImage *procimg = cvCloneImage(gray);


	// Normalization
	IplImage* size_img = cvCreateImage( cvSize( DEST_WIDTH,DEST_HEIGHT),8,1);// Set a uniform size
	cvResize( gray, size_img, CV_INTER_LINEAR );// Resize the gray-scale image

	imageClone(size_img,outimage);// Copy the image to *outimage

	cvReleaseImage(&gray);
	cvReleaseImage(&size_img);
	cvReleaseImage(&procimg);
}

// Compare two strings for sorting
// Because the original file names of MORPH database are not arranged in ascending order of corresponding age
bool smaller(const string & str1, const string & str2) 
{
	// Find the corresponding sequence number of string 1 
	int pos1 = str1.rfind ("_"); // find the last "_"
	if (pos1 == string::npos)
	{
		return false;
	}

	int pos2 = str1.rfind ("M"); // Find the last "M" = Male
	if (pos2 == string::npos)// If no, we may find "F" = Female
	{
		pos2 = str1.rfind ("F"); 
		if (pos2 == string::npos)
		{
			return false;
		}
	}

	string strindex1 = str1.substr(pos1+1,pos2-pos1-1);// Obtain the string including age
	int index1 = atoi(strindex1.c_str());// Convert to integer


	// Find the corresponding age of string 2
	pos1 = str2.rfind ("_"); 
	if (pos1 == string::npos)
	{
		return false;
	}

	pos2 = str2.rfind ("M");
	if (pos2 == string::npos)
	{
		pos2 = str2.rfind ("F"); 
		if (pos2 == string::npos)
		{
			return false;
		}
	}

	string strindex2 = str2.substr(pos1+1,pos2-pos1-1);
	int index2 = atoi(strindex2.c_str());

	return index1<index2;
}


// This function can create mapping table. 256 bins to 59 bins
void CreateMap(uchar maptable[])
{
	int i;
	for (i=0;i<256;i++)
	{
		maptable[i] = 58;// Initialize to 58
	}
	
	for (i=0;i<58;i++)
	{
		int index = LBP_NUM[i];// Obtain corresponding position
		maptable[index] = i;// Assign the value
	}
}


// This function can calculate LBP histogram.
void CALC_LBPHIST(Mat image,float result[],uchar maptable[])
{
	memset(result,0,sizeof(float)*59);
	for (int i=0;i<image.rows;i++)
	{
		for (int j=0;j<image.cols;j++)
		{
			int ilbpvalue = image.at<uchar>(i,j);// Pixel Value
			uchar dimvalue = maptable[ilbpvalue];// Transform it to values ranging from 0~58 according to the mapping table

			result[dimvalue] = result[dimvalue]+1;
		}
	}
	for (int i=0;i<59;i++)
	{
		result[i] = result[i]/(image.rows*image.cols);// Normalization
	}
}

// This function can calculate LBP features for each region.
Mat COMPUTE_BLOCKLBP(Mat image,Mat &result,uchar maptable[])
{
	int totalblocknum = yblocknum*xblocknum;// Total block numbers
	result = Mat(totalblocknum,59,CV_32FC1,Scalar::all(0));// LBP features - a matrix of totalblocknum * 59

	Mat lbpimg = image.clone();
	uchar center=0;// Original center pixel value
	uchar center_lbp=0;// LBP label for that pixel
	int row,col;
	// Traverse the image. For each pixel, threshold the 3¡Á3 neighborhood with the center pixel value.
	// Represent the result as binary number and convert it to decimal value.
	for (int row=1; row<image.rows-1; row++)
	{
		for (int col=1; col<image.cols-1; col++)
		{
			center = image.at<uchar>(row,col);// Pixel value
			center_lbp = 0;

			uchar tmpvalue =  image.at<uchar>(row-1,col-1);
			if(center <= tmpvalue)
			{
				center_lbp += 128;
			}

			tmpvalue =  image.at<uchar>(row-1,col);
			if(center <= tmpvalue)
			{
				center_lbp += 64;
			}


			tmpvalue =  image.at<uchar>(row-1,col+1);
			if(center <= tmpvalue)
			{
				center_lbp += 32;
			}

			tmpvalue =  image.at<uchar>(row,col+1);
			if(center <= tmpvalue)
			{
				center_lbp += 16;
			}

			tmpvalue =  image.at<uchar>(row+1,col+1);
			if(center <= tmpvalue)
			{
				center_lbp += 8;
			}

			tmpvalue =  image.at<uchar>(row+1,col);
			if(center <= tmpvalue)
			{
				center_lbp += 4;
			}

			tmpvalue =  image.at<uchar>(row+1,col-1);
			if(center <= tmpvalue)
			{
				center_lbp += 2;
			}

			tmpvalue =  image.at<uchar>(row,col-1);
			if(center <= tmpvalue)
			{
				center_lbp += 1;
			}

			lbpimg.at<uchar>(row,col) = center_lbp;// LBP label

		}
	}
	// Obtain the LBP image "lbpimg"

	// Divide the LBP image and calculate the 59-bin LBP histogram for each region
	Rect r;// Region of interest (ROI)
	r.width = int(DEST_WIDTH/xblocknum);
	r.height = int(DEST_HEIGHT/yblocknum);
	
	int blockindex = 0;
	for (int i=0;i<yblocknum;i++)
	{
		for (int j=0;j<xblocknum;j++)
		{
			r.y = i*r.height;
			r.x = j*r.width;

			Mat blockimg = lbpimg(r);// Obtain rectangular ROI r

			float hist[59] = {0};
			CALC_LBPHIST(blockimg,hist,maptable);// Calculate the 59-bin LBP histogram for the region

			float *data = result.ptr<float>(i);// Pointer of each row
			memcpy(data,hist,sizeof(float)*59);// Assign the value to the result
		}
	}
	
	
	return lbpimg;
}

// This function can calculate the weighted Chi square distance.
float computedist(Mat feature1,Mat feature2)
{
	float dist = 0;
	int totalblocknum = yblocknum*xblocknum;// Total partition number
	for (int i=0;i<totalblocknum;i++)
	{
		float blockdist = 0;// The distance between blocks
		for(int j=0;j<59;j++)
		{
			float f1 = (float)feature1.at<float>(i,j);
			float f2 = (float)feature2.at<float>(i,j);
			float d=0;
			if (f1==0 && f2==0)// Prevent division errors
			{
				d=0;
			}
			else
			{
				
				d = (f1-f2)*(f1-f2)/(f1+f2);// Chi square distance
			}

			blockdist = blockdist+d;
			
		}
		
		// Use the weights
		const float *w = weight[0];
		dist = w[i]*blockdist + dist;// The weighted Chi square distance
	}

	return dist ;
}

// This function can calculate the Euclidean distance.
float computedist2(Mat feature1,Mat feature2)
{
	float dist = 0;
	int totalblocknum = yblocknum*xblocknum;
	for (int i=0;i<totalblocknum;i++)
	{
		float blockdist = 0;
		for(int j=0;j<59;j++)
		{
			float f1 = (float)feature1.at<float>(i,j);
			float f2 = (float)feature2.at<float>(i,j);
			blockdist = blockdist+(f1-f2)*(f1-f2);
		}
		
		blockdist = sqrt(blockdist);

		const float *w = weight[0];
		dist = w[i]*blockdist + dist;
	}

	return dist ;
}

// This function can test a single image with datapath.
void singletest()
{
	char strfilename[260] = {0};// The filename of testing image
	printf("Please input the datapath of the testing image:\r\n");
	scanf("%s",strfilename);

	IplImage *image = cvLoadImage(strfilename);// Read the image
	if(!image)
	{
		printf("Load failed. Please make sure that your input is a valid path.\r\n");
		return;
	}
	IplImage* imageproc = NULL;// The preprocessed image
	preproc(image,&imageproc);// Preprocessing
	Mat img(imageproc);
	Mat lbpfeature;
	COMPUTE_BLOCKLBP(img,lbpfeature,maptable);// Calculate LBP features for each region

	float minvalue =DBL_MAX;
	int mintype = -1;
	float dist = 0;

	int alltrainpicnum = trainfeaturelist.size();// The number of all training samples

	for(int i=0;i<alltrainpicnum;i++)// Traverse all images
	{
		Mat tmp = trainfeaturelist[i];
		float dist = computedist(lbpfeature,tmp);// Calculate distance
		if(dist < minvalue)// Find the nearest neighbor and its index
		{
			minvalue = dist;
			mintype = trainlabels[i];
		}
	}

	printf("%s is recognized as person %d\r\n",strfilename,mintype);
	
	cvReleaseImage(&image);
	cvReleaseImage(&imageproc);
}

// This function can test all testing images and give the recognition rate.
void alltest()
{
	int testnum = 0;
	int rightnum = 0;


	int i,j;
	char strpath[260] = {0};// Subfolder path
	for (i=1;i<=PEOPLENUM;i++)// Traverse every subject  
	{

		sprintf_s(strpath,260,"MORPH_PROC\\%d",i);// The folder containing preprocessed MORPH database
		vector<string> files;
		vector<string> filenames;
		getPicFiles(strpath,files,filenames);// Obtain the number of picture files in subfolders
		sort(files.begin(),files.end(),smaller);// Order the files according to "smaller" function
		sort(filenames.begin(),filenames.end(),smaller);// Order the filenames according to "smaller" function 

		int filenum = files.size();
		for(j=filenum-1;j<filenum;j++)//// Traverse all testing images in every subfolder
		{
			IplImage *image = cvLoadImage(files[j].c_str(),1);// Read image
			if(!image)
			{
				continue;
			}
			IplImage* imageproc = NULL;
			preproc(image,&imageproc);
			Mat img(imageproc);
			Mat lbpfeature;
			COMPUTE_BLOCKLBP(img,lbpfeature,maptable);// Calculate LBP features for each region
			cvReleaseImage(&image);
			cvReleaseImage(&imageproc);

			float minvalue =DBL_MAX;
			int mintype = -1;
			float dist = 0;

			int alltrainpicnum = trainfeaturelist.size();// The number of all training samples

			for(int k=0;k<alltrainpicnum;k++)// Traverse all images
			{
				Mat tmp = trainfeaturelist[k];
				float dist = computedist(lbpfeature,tmp);// Calculate the weighted Chi square distance
			//	float dist = computedist2(lbpfeature, tmp);// Calculate the Euclidean distance
				if(dist < minvalue)// Find the nearest neighbor and its index
				{
					minvalue = dist;
					mintype = trainlabels[k];
				}
			}

			if(mintype == i)// If the index of the nearest neighbor is same as testing image£¬then the recognition is correct
			{
				rightnum++;
				printf("%s is recognized as person %d, correct\r\n",files[j].c_str(),mintype);
			}
			else
			{
				printf("%s is recognized as person %d, wrong\r\n",files[j].c_str(),mintype);
			}
			
			
			testnum++;

		}
		//cvReleaseImage(&image);

	}
	
	float rightrate = float(rightnum)/float(testnum);// 1st rank recognition rate
	printf("Tested %d people.\r\n%d people are correctly classified.\r\nThe recognition rate is %.2f%%\r\n",testnum,rightnum,rightrate*100);
}


void main()
{
	trainfeaturelist.reserve(100);
	int aaa = trainfeaturelist.size();
	CreateMap(maptable);// Create mapping table
	
	printf("Training begin.....\r\n");
	int i,j;
	char strpath[260] = {0};// Subfolder path
	for (i=1;i<=PEOPLENUM;i++)// Traverse every subject 
	{
		
		printf("Training person %d\r\n",i);
		sprintf_s(strpath,260,"MORPH_PROC\\%d",i);// The folder containing preprocessed MORPH database
		vector<string> files;
		vector<string> filenames;
		getPicFiles(strpath,files,filenames);
		sort(files.begin(),files.end(),smaller);
		sort(filenames.begin(),filenames.end(),smaller); 
		
		for(j=0;j<TRAINNUM;j++)// Traverse all training images of the subjects
		{
			IplImage *image = cvLoadImage(files[j].c_str(),1);
			if(!image)
			{
				continue;
			}
			IplImage* imageproc = NULL;
			preproc(image,&imageproc);
			Mat img(imageproc);
			Mat lbpfeature;
			COMPUTE_BLOCKLBP(img,lbpfeature,maptable);
			trainfeaturelist.push_back(lbpfeature);// Store the LBP feature into "trainfeaturelist"
			
			trainlabels.push_back(i);
			cvReleaseImage(&image);
			cvReleaseImage(&imageproc);
		}
		//cvReleaseImage(&image);
		
	}

	printf("Training complete. Now you can do the test.\r\n");
	
	while (1)
	{
		printf("\r\n***********************************\r\n");
		printf("Please enter a number:\r\n");
		printf("1.Single Test\r\n");
		printf("2.All Test\r\n");
		printf("3.Exit\r\n");

		int selectnum= 0;
		scanf("%d",&selectnum);

		if (selectnum == 1)
		{
			singletest();
		}

		if (selectnum == 2)
		{
			alltest();
		}

		if (selectnum == 3)
		{
			break;
		}
	}
	
	printf("Thank you for using the LBP program!\r\n");
	getchar();
}

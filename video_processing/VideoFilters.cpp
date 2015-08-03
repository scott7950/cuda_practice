/******************************************************************************
*
*            (C) Copyright 2014 The Board of Trustees of the
*                        Florida Institute of Technology
*                         All Rights Reserved
*
* Lab Image Filters
******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <iostream>

/*CUDA*/
#include <cuda_runtime.h>

/*OpenCV*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "VideoFilters.h"

using namespace std;

//Calculate blocks
int iDivUp(int n, int blockDim){
	int nBlocks = n / blockDim;
	if (n % blockDim)
		return nBlocks++;

	return nBlocks;
}

void usage(){
	printf("\n    Invalid input parameters!"
		"\n    Usage: .\\lab-ImageFilters.exe                "
		"\n    Usage: .\\lab-ImageFilters.exe <video file name>            "
		"\n");

	exit(0);
}

//PARAMETERS
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

int main(int argc, char **argv){


	string name_imag;
	if (argc == 1){
		name_imag.assign("Wildlife.mp4");
	}
	else if (argc == 2){
		cout << "Input FileName:  " << argv[1] << endl;
		name_imag.assign(argv[1]);
	}
	else{
		usage();
	}


	cv::VideoCapture cap(name_imag);  //open the default camera
	
	/* Check to see if the video capture opened successful */
	if (!cap.isOpened())  //check if we succeeded
		return -1;

	/* Gather some video statistics */
	double width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	double height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	double TotalFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);

	/* Declare video display windows */
	cv::namedWindow("Original Video", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Mean Filtered Video", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Gaussian Filtered Video", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Laplacian Filtered Gaussian Video", CV_WINDOW_AUTOSIZE);

	cv::Mat frame_Original;   //Original Image

	int fNumber = 0;
	while (cap.isOpened())
	{

		/* Load image into Host Variables */
		cap >> frame_Original;

		///***************************************************
		//*    Mean Filter
		//***************************************************/

		/*Size of Image*/
		int imageW = frame_Original.cols;
		int imageH = frame_Original.rows;

		size_t size = imageW*imageH*sizeof(uchar4);

		/* Create a 4 channel data structures */
		cv::Mat frame_Original4c_median;
		frame_Original4c_median.create(imageH, imageW, CV_8UC(4));
		cv::cvtColor(frame_Original, frame_Original4c_median, CV_BGR2BGRA, 0);

		/* Create device Variables & device memory */
		uchar4 *Image_dev;
		cudaMalloc((void **)&Image_dev, size);
		CUDA_CreateMemoryArray(imageW, imageH);

		/*Copy Memory (Host-->Device)*/
		cudaMemcpy(Image_dev, frame_Original4c_median.data, size, cudaMemcpyHostToDevice);

		/*Define the size of the grid and thread blocks*/
		dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, 1);
		dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y), 1);

		CUDA_MemcpyToArray(Image_dev, imageW, imageH);
		CUDA_BindTextureToArray();

		/* Mean Filter Launch the Kernel Function*/
		CUDA_MeanFilter(Image_dev, imageW, imageH, grid, threads);

		/*Copy Memory (Device-->Host)*/
		cudaMemcpy(frame_Original4c_median.data, Image_dev, size, cudaMemcpyDeviceToHost);


		/*Device*/
		cudaFree(Image_dev);
		CUDA_FreeArrays();


		/***************************************************
		*    Gaussian Filter
		***************************************************/

		/* Create a 4 channel data structures */
		cv::Mat frame_Original4c_gauss;
		frame_Original4c_gauss.create(imageH, imageW, CV_8UC(4));
		cv::cvtColor(frame_Original, frame_Original4c_gauss, CV_BGR2BGRA, 0);

		/* Create device Variables & device memory */
		cudaMalloc((void **)&Image_dev, size);
		CUDA_CreateMemoryArray(imageW, imageH);

		/*Copy Memory (Host-->Device)*/
		cudaMemcpy(Image_dev, frame_Original4c_gauss.data, size, cudaMemcpyHostToDevice);

		CUDA_MemcpyToArray(Image_dev, imageW, imageH);
		CUDA_BindTextureToArray();

		/* Gaussian Filter Launch the Kernel Function*/
		CUDA_GaussianFilter(Image_dev, imageW, imageH, grid, threads);

		/*Copy Memory (Device-->Host)*/
		cudaMemcpy(frame_Original4c_gauss.data, Image_dev, size, cudaMemcpyDeviceToHost);


		/*Device*/
		cudaFree(Image_dev);
		CUDA_FreeArrays();

		/***************************************************
		*    Laplacian Filter
		***************************************************/

		cv::Mat frame_Original1c;

		size_t sizef = imageW*imageH*sizeof(float);
		frame_Original1c.create(imageH, imageW, CV_32FC(1));

		/* Create device Variables & device memory */
		float  *Imagef_dev;
		cudaMalloc((void **)&Imagef_dev, sizef);
		cudaMalloc((void **)&Image_dev, size);
		CUDA_CreateMemoryArray(imageW, imageH);

		/*Copy Memory (Host-->Device)*/
		//cudaMemcpy(Image_dev, frame_Original4c_gauss.data, size, cudaMemcpyHostToDevice);
		cudaMemcpy(Image_dev, frame_Original4c_median.data, size, cudaMemcpyHostToDevice);

		CUDA_MemcpyToArray(Image_dev, imageW, imageH);
		CUDA_BindTextureToArray();

		/* Laplacian Filter Launch the Kernel Function*/
		CUDA_LaplacianFilter(Imagef_dev, imageW, imageH, grid, threads);

		/*Copy Memory (Device-->Host)*/
		cudaMemcpy(frame_Original1c.data, Imagef_dev, sizef, cudaMemcpyDeviceToHost);
		cv::normalize(frame_Original1c, frame_Original1c, 255, 0);


		/*Device*/
		cudaFree(Imagef_dev);
		cudaFree(Image_dev);
		CUDA_FreeArrays();

		/*  Conditional checks for data and bounds of video*/

		if (frame_Original.data != NULL)
		{
			cv::imshow("Original Video", frame_Original);
		}

		if (frame_Original4c_median.data != NULL)
		{
			cv::imshow("Mean Filtered Video", frame_Original4c_median);
		}

		if (frame_Original4c_gauss.data != NULL)
		{
			cv::imshow("Gaussian Filtered Video", frame_Original4c_gauss);
		}

		if (frame_Original4c_gauss.data != NULL)
		{
			cv::imshow("Laplacian Filtered Gaussian Video", frame_Original1c);
		}

		/*  Checks for a video capture stall */
		if (cv::waitKey(30) >= 0) break;

		if (fNumber > TotalFrames) break;
		fNumber++;

	}//End while (cap.isOpened())



	cv::destroyWindow("Mean Filtered Video");
	cv::destroyWindow("Original Video");
	cv::destroyWindow("Gaussian Filtered Video");
	cv::destroyWindow("Laplacian Filtered Gaussian Video");


	return(0);
}

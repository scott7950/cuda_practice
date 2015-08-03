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

#include "ImageFilters.h"

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
		"\n    Usage: .\\lab-ImageFilters.exe <m>            "
		"\n");

	exit(0);
}

//PARAMETERS
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

int main(int argc, char **argv){


	string name_imag;
	if (argc == 1){
		name_imag.assign("Estacion.jpg");
	}
	else if (argc == 2){
		cout << "Input FileName:  " << argv[1] << endl;
		name_imag.assign(argv[1]);
	}
	else{
		usage();
	}

	/* Load image into Host Variables */
	cv::Mat frame_Original;   //Original Image
	frame_Original = cv::imread(name_imag, 1);

	if (frame_Original.data == NULL)
	{
		printf("Could not open or find the image\n");
		exit(1);
	}

	/***************************************************
	*    Mean Filter
	***************************************************/
	cv::namedWindow("Original Frame", CV_WINDOW_AUTOSIZE);
	cv::imshow("Original Frame", frame_Original);
	cv::waitKey(0);

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

	cv::namedWindow("Mean Filtered Frame", CV_WINDOW_AUTOSIZE);
	cv::imshow("Mean Filtered Frame", frame_Original4c_median);
	cv::waitKey(0);


	/*Device*/
	cudaFree(Image_dev);
	CUDA_FreeArrays();



	cv::destroyWindow("Mean Filtered Frame");
	cv::destroyWindow("Original Frame");

	return(0);
}

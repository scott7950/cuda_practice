/******************************************************************************
*
*            (C) Copyright 2014 The Board of Trustees of the
*                        Florida Institute of Technology
*                         All Rights Reserved
*
* Lab Image Filters
******************************************************************************/
#include "VideoFilters.h"

//CUDA 
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_string.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>

/*TEXTURES*/
texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage_rgb;


cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();

cudaArray *cu_image;

extern "C"
void CUDA_CreateMemoryArray(int imageW,int imageH){
	cudaMallocArray(&cu_image, &uchar4tex, imageW, imageH);
}

extern "C"
void CUDA_BindTextureToArray(){
	cudaBindTextureToArray(texImage_rgb,cu_image);
}

extern "C"
void CUDA_FreeArrays(){
	cudaFreeArray(cu_image);
}

extern "C"
void CUDA_MemcpyToArray(uchar4 *src,int imageW,int imageH){
	cudaMemcpyToArray( cu_image, 0, 0,src, imageW * imageH * sizeof(uchar4), cudaMemcpyDeviceToDevice);
}

/***************************************
	Box Filter
*****************************************/

__constant__ float MeanKernel[9] = {1,1,1,  
                                    1,1,1,
									1,1,1};

/***************************************
	Median Filter Kernel Function
*****************************************/
__global__ void MeanFilter(uchar4 *Image_dev, int w, int h){
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    const float x=(float)ix+0.5f;
    const float y=(float)iy+0.5f;

	int win_W=1;

	if(ix < w && iy < h){
		float4 pixval;
		float3 sum;
		sum.x=0.0f;
		sum.y=0.0f;
		sum.z=0.0f;
		int k=0;
		for(int ii=-win_W;ii<=win_W;ii++){
			for(int jj=-win_W;jj<=win_W;jj++){
				pixval=tex2D(texImage_rgb,x+ii,y+jj);
				sum.x+=pixval.x*MeanKernel[k];
				sum.y+=pixval.y*MeanKernel[k];
				sum.z+=pixval.z*MeanKernel[k];
				k++;
			}
		}
		Image_dev[w*iy+ix].x=(unsigned char)((sum.x/9)*255);
		Image_dev[w*iy+ix].y=(unsigned char)((sum.y/9)*255);
		Image_dev[w*iy+ix].z=(unsigned char)((sum.z/9)*255);
	}
}

/***************************************
	Median Filter Calling Function
*****************************************/
extern "C"
void CUDA_MeanFilter(uchar4 *Image_dev,int imageW,int imageH,dim3 grid,dim3 threads){
	MeanFilter<<<grid,threads>>>(Image_dev,imageW,imageH);
}

/***************************************
 *   Gaussian Filter
*****************************************/
__constant__ float GaussianKernel[9] = {1,2,1,  
                                        2,4,2,
									    1,2,1};

__constant__ float GaussianKernel55[25] = {1,4,7,4,1,
										   4,16,26,16,4,
										   7,26,41,26,7,
										   4,16,26,16,4,
										   1,4,7,4,1};

/***************************************
	Gaussian Filter Kernel Function
*****************************************/
__global__ void GaussianFilter(uchar4 *Image_dev, int w, int h){
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	const float x = (float)ix + 0.5f;
	const float y = (float)iy + 0.5f;

	//int win_W = 1;
	int win_W = 2;
	//int normalizingFactor = 16;
	int normalizingFactor_55 = 273;

	if (ix < w && iy < h){
		float4 pixval;
		float3 sum;
		sum.x = 0.0f;
		sum.y = 0.0f;
		sum.z = 0.0f;
		int k = 0;
		for (int ii = -win_W; ii <= win_W; ii++){
			for (int jj = -win_W; jj <= win_W; jj++){
				pixval = tex2D(texImage_rgb, x + ii, y + jj);
				//sum.x += pixval.x*GaussianKernel[k];
				//sum.y += pixval.y*GaussianKernel[k];
				//sum.z += pixval.z*GaussianKernel[k];
				sum.x += pixval.x*GaussianKernel55[k];
				sum.y += pixval.y*GaussianKernel55[k];
				sum.z += pixval.z*GaussianKernel55[k];
				k++;
			}
		}
		//Image_dev[w*iy + ix].x = (unsigned char)((sum.x / normalizingFactor) * 255);
		//Image_dev[w*iy + ix].y = (unsigned char)((sum.y / normalizingFactor) * 255);
		//Image_dev[w*iy + ix].z = (unsigned char)((sum.z / normalizingFactor) * 255);
		Image_dev[w*iy + ix].x = (unsigned char)((sum.x / normalizingFactor_55) * 255);
		Image_dev[w*iy + ix].y = (unsigned char)((sum.y / normalizingFactor_55) * 255);
		Image_dev[w*iy + ix].z = (unsigned char)((sum.z / normalizingFactor_55) * 255);
	}
}

/***************************************
	Gaussian Filter Calling Function
*****************************************/
extern "C"
void CUDA_GaussianFilter(uchar4 *Image_dev, int imageW, int imageH, dim3 grid, dim3 threads){
	GaussianFilter << <grid, threads >> >(Image_dev, imageW, imageH);
}


/***************************************
*   Laplacian Filter
*****************************************/

__constant__ float LaplacianKernel[9] = { 0, 1, 0,
										1, -4, 1,
										0, 1, 0 };

/***************************************
	Laplacian Filter Kernel Function
*****************************************/
__global__ void LaplacianFilter(float *Image_dev, int w, int h){
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	const float x = (float)ix + 0.5f;
	const float y = (float)iy + 0.5f;

	int win_W = 1;

	if (ix < w && iy < h){
		float4 pixval;
		float sum = 0.0f;

		int k = 0;
		for (int ii = -win_W; ii <= win_W; ii++){
			for (int jj = -win_W; jj <= win_W; jj++){
				pixval = tex2D(texImage_rgb, x + ii, y + jj);
				float Intensidad = (pixval.x + pixval.y + pixval.z) / 3.0f;

				sum += Intensidad*LaplacianKernel[k];

				k++;
			}
		}
		Image_dev[w*iy + ix] = sum;
	}
}

/***************************************
	Laplacian Filter Calling Function
*****************************************/
void CUDA_LaplacianFilter(float *Image_dev, int imageW, int imageH, dim3 grid, dim3 threads){
	LaplacianFilter << <grid, threads >> >(Image_dev, imageW, imageH);
}

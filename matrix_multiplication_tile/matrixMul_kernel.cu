/******************************************************************************
*
*            (C) Copyright 2014 The Board of Trustees of the
*                        Florida Institute of Technology
*                         All Rights Reserved
*
* Lab 2 Matrix Multiplication
******************************************************************************/
// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <iostream>

//CUDA runtime
#include <cuda_runtime.h>

//Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>

using namespace std;

#define TILE_WIDTH 16

void verify(float *A_h, float *B_h, float *C_h, int width);

__global__ void matrixMulKernel(int width, float *A_d, float *B_d, float* C_d) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (width x width) matrix
     *   where B is a (width x width) matrix
     *   where C is a (width x width) matrix
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
#ifdef USE_TILE
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

	if ((row < width) && (col < width)){
        float Pvalue = 0;

        for(int m=0; m<width/TILE_WIDTH; ++m) {
            Mds[ty][tx] = d_M[row*width + m*TILE_WIDTH + tx];
            Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*width + col];
            __syncthreads();

            for(int k=0; k<TILE_WIDTH; ++k) {
                Pvalue += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();

        }

        C_d[row*width + col] = Pvalue;
    }

#else
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if ((row < width) && (col < width)){
		float Pvalue = 0;
		for (int k = 0; k < width; k++){
			Pvalue += A_d[row*width + k] * B_d[k*width+col];
		}
		C_d[row*width + col] = Pvalue;
	}
#endif
}

extern "C" void
matrixMultiplicationFunction(int width)
{
	clock_t beginTotalTime;
	clock_t endTotalTime;
	clock_t begin;
	clock_t end;

	beginTotalTime = clock();

	printf("\nSetting up the problem...\n"); fflush(stdout);
	begin = clock();

	/* Matrix size in bytes */

	/* Calculate the data size for the matrix allocation */
	int nElements = width*width;

	/* Initialize the matrices with random data */
	
	// Random data for host array A
	float* A_h = (float*)malloc(nElements*sizeof(float));
	for (unsigned int i = 0; i < nElements; i++) {
		A_h[i] = (rand() % 100) / 100.00; 
	}


	//Random data for host array B
	float* B_h = (float*)malloc(nElements*sizeof(float));
	for (int i = 0; i < nElements; i++) {
		B_h[i] = (rand() % 100) / 100.00; 
	}

	float* C_h = (float*)malloc(nElements*sizeof(float));


	end = clock();
	printf("Elapsed: %f seconds\n", (double)(end - begin) / CLOCKS_PER_SEC);

	printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", width, width,
		width, width, width, width);

	// Allocate device variables ----------------------------------------------
	printf("Allocating device variables...\n"); fflush(stdout);
	begin = clock();


	//INSERT CODE HERE
	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, nElements*sizeof(float));

	cudaMalloc((void **)&d_B, nElements*sizeof(float));

	cudaMalloc((void **)&d_C, nElements*sizeof(float));



	checkCudaErrors(cudaDeviceSynchronize());
	end = clock();
	printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

	// Copy host variables to device ------------------------------------------
	printf("Copying data from host to device...\n"); fflush(stdout);
	begin = clock();

	//INSERT CODE HERE
	cudaMemcpy(d_A, A_h, nElements*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B_h, nElements*sizeof(float), cudaMemcpyHostToDevice);




	checkCudaErrors(cudaDeviceSynchronize());
	end = clock();
	printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

	// Launch kernel using standard sgemm interface ---------------------------
	printf("Launching kernel...\n"); fflush(stdout);
	begin = clock();

	// Initialize thread block and kernel grid dimensions ---------------------

	int BLOCK_SIZE = 16; // Use 16x16 thread blocks

	//INSERT CODE HERE

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 dimGrid(ceil(width / (float)BLOCK_SIZE), ceil(width / (float)BLOCK_SIZE), 1);

	// Invoke CUDA kernel -----------------------------------------------------

	//INSERT CODE HERE
	matrixMulKernel <<<dimGrid, dimBlock >>>(width, d_A, d_B, d_C);


	checkCudaErrors(cudaDeviceSynchronize());
	end = clock();
	printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

	// Copy device variables from host ----------------------------------------
	printf("Copying data from device to host...\n"); fflush(stdout);
	begin = clock();

	//INSERT CODE HERE
	cudaMemcpy(C_h, d_C, nElements*sizeof(float), cudaMemcpyDeviceToHost);


	checkCudaErrors(cudaDeviceSynchronize());
	end = clock();
	printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);
	endTotalTime = clock();

	// Verify correctness -----------------------------------------------------

	printf("Verifying results...\n"); fflush(stdout);

	verify(A_h, B_h, C_h, width);

	// Free memory ------------------------------------------------------------

	free(A_h);
	free(B_h);
	free(C_h);

	//INSERT CODE HERE
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);


	printf("Total Elapsed Processing: %f seconds\n", (double)(endTotalTime - beginTotalTime) / CLOCKS_PER_SEC);

}


void verify(float *A_h, float *B_h, float *C_h, int width) {

	const float relativeTolerance = 1e-6;

	for (int row = 0; row < width; ++row) {
		for (int col = 0; col < width; ++col) {
			float sum = 0;

			for (int i = 0; i < width; ++i) {
				sum += A_h[row*width + i] * B_h[i*width + col];
			}
			

			float relativeError = (sum - C_h[row*width + col]) / sum;
			if (relativeError > relativeTolerance
				|| relativeError < -relativeTolerance) {
				printf("TEST FAILED\n\n");
				exit(0);
			}
		}
	}

	printf("TEST PASSED\n\n");

}

extern "C" void
matrixMultiplication_C(int width) {
	
	clock_t beginTotalTime;
	clock_t endTotalTime;
	clock_t begin;
	clock_t end;

	beginTotalTime = clock();

	printf("\nSetting up the problem...\n"); fflush(stdout);
	begin = clock();

	/* Matrix size in bytes */

	/* Calculate the data size for the matrix allocation */
	int nElements = width*width;
	int dataMemorySize = nElements * sizeof(float);

	/* Initialize the matrices with random data */

	// Random data for host array A
	float* A_h = (float*)malloc(dataMemorySize);
	for (unsigned int i = 0; i < nElements; i++) {
		A_h[i] = (rand() % 100) / 100.00;
	}


	//Random data for host array B
	float* B_h = (float*)malloc(dataMemorySize);
	for (int i = 0; i < nElements; i++) {
		B_h[i] = (rand() % 100) / 100.00;
	}

	float* C_h = (float*)malloc(dataMemorySize);


	end = clock();
	printf("Elapsed: %f seconds\n", (double)(end - begin) / CLOCKS_PER_SEC);

	printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", width, width,
		width, width, width, width);

	printf("Launch C processing...\n"); fflush(stdout);
	begin = clock();

	for (int row = 0; row < width; ++row) {
		for (int col = 0; col < width; ++col) {
			float sum = 0;

			for (int i = 0; i < width; ++i) {
				sum += A_h[row*width + i] * B_h[i*width + col];
			}

			C_h[row*width + col] = sum;

		}
	}

	end = clock();
	printf("Elapsed: %f seconds\n", (double)(end - begin) / CLOCKS_PER_SEC);
	endTotalTime = clock();

	printf("Verifying results...\n"); fflush(stdout);

	verify(A_h, B_h, C_h, width);

	// Free memory ------------------------------------------------------------

	free(A_h);
	free(B_h);
	free(C_h);

	printf("Total Elapsed Processing: %f seconds\n", (double)(endTotalTime - beginTotalTime) / CLOCKS_PER_SEC);
}

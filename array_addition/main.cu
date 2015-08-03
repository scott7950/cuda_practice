/******************************************************************************
 *		Lab 1
 *
 ******************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

void verify(float *A, float *B, float *C, int n);

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {

    // Calculate global thread index based on the block and thread indices ----

    //INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockDim.x* blockIdx.x;

    // Use global index to determine which elements to read, add, and write ---

    //INSERT KERNEL CODE HERE
    if (i<n)
        C[i] = A[i] + B[i];
}

int main(int argc, char**argv) {
    clock_t begin, end;

    // Initialize host variables ----------------------------------------------
    printf("\nSetting up the problem...\n"); fflush(stdout);
    begin = clock();

    unsigned int n;
    if (argc == 1) {
        n = 10000;
    }
    else if (argc == 2) {
        n = atoi(argv[1]);
    }
    else {
        printf("\n    Invalid input parameters!"
                "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
                "\n    Usage: ./vecadd <m>           # Vector of size m is used"
                "\n");
        exit(EXIT_SUCCESS);
    }

    float* A_h = (float*)malloc(sizeof(float)*n);
    for (unsigned int i = 0; i < n; i++) { A_h[i] = (rand() % 100) / 100.00; }

    float* B_h = (float*)malloc(sizeof(float)*n);
    for (unsigned int i = 0; i < n; i++) { B_h[i] = (rand() % 100) / 100.00; }

    float* C_h = (float*)malloc(sizeof(float)*n);

    end = clock();
    printf("Elapsed: %f seconds\n", (double)(end - begin) / CLOCKS_PER_SEC);
    printf("    Vector size = %u\n\n", n);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables...\n"); fflush(stdout);
    begin = clock();

    //INSERT CODE HERE
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeof(float)*n);
    cudaMalloc((void **)&d_B, sizeof(float)*n);
    cudaMalloc((void **)&d_C, sizeof(float)*n);

    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device...\n"); fflush(stdout);
    begin = clock();

    //INSERT CODE HERE
    cudaMemcpy(d_A, A_h, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_h, sizeof(float)*n, cudaMemcpyHostToDevice);

    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel...\n"); fflush(stdout);
    begin =clock();

    //INSERT CODE HERE
    vecAddKernel <<<ceil(n / 256.0), 256 >>>(d_A, d_B, d_C, n);

    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host...\n"); fflush(stdout);
    begin = clock();

    //INSERT CODE HERE
    cudaMemcpy(C_h, d_C, sizeof(float)*n, cudaMemcpyDeviceToHost);

    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

    // Verify correctness -----------------------------------------------------

    printf("Verifying results...\n"); fflush(stdout);

    verify(A_h, B_h, C_h, n);

    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    exit(EXIT_SUCCESS);
}

void verify(float *A, float *B, float *C, int n) {

    const float relativeTolerance = 1e-6;

    for (int i = 0; i < n; i++) {
        float sum = A[i] + B[i];
        float relativeError = (sum - C[i]) / sum;
        if (relativeError > relativeTolerance
                || relativeError < -relativeTolerance) {
            printf("TEST FAILED\n\n");
            exit(0);
        }
    }
    printf("TEST PASSED\n\n");

}


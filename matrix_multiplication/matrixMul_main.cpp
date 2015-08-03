/******************************************************************************
 *
 *            (C) Copyright 2014 The Board of Trustees of the
 *                        Florida Institute of Technology
 *                         All Rights Reserved
 *
 * Lab 2 Matrix Multiplication
 ******************************************************************************/
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace std;

extern "C" void matrixMultiplicationFunction(int width);
extern "C" void matrixMultiplication_C(int width);

int main (int argc, char *argv[])
{

    // Initialize host variables ----------------------------------------------

	int width;

	if (argc == 1) {
		width = 1000;
	}
	else if (argc == 2) {
		width = atoi(argv[1]);
	}
	else {
		printf("\n    Invalid input parameters!"
			"\n    Usage: ./sgemm                # All matrices are 1000 x 1000"
			"\n    Usage: ./sgemm <m>            # All matrices are m x m"
			"\n");
		exit(0);
	}

	for (int i = 0; i < 3; i++) {
		printf("Process C Matrix Multiplication");
		matrixMultiplication_C(width / (i+1));

		printf("\n\n\n");

		printf("Process GPU Matrix Multiplication");
		matrixMultiplicationFunction(width / (i+1));
	}

    return 0;

}


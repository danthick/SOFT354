#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <cstddef>
#include <stddef.h>
#include "Matrix.cu"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning(disable:4996)


// Thread block size 
#define BLOCK_SIZE 75

__device__ float GetElementMatrixA(const Matrix A, int m, int block_row, int row, int col) {
	/*
		When the matrix dimension is not- multiple of the tile dimensions,
		then some tiles will only partially overlap the matrice.
	*/

	if ((m * BLOCK_SIZE + col < A.width) && (block_row * BLOCK_SIZE + row < A.height))
	{
		return A.elements[(block_row * BLOCK_SIZE + row) * A.width + m * BLOCK_SIZE + col];
	}
	else
	{
		/* The elements of the tiles partially overlapping the matrice are set to zero (padding) */
		return 0.0;
	}
}

__device__ float GetElementMatrixB(const Matrix B, int m, int block_col, int row, int col) {
	/*
		When the matrix dimension is not- multiple of the tile dimensions,
		then some tiles will only partially overlap the matrice.
	*/
	if (m * BLOCK_SIZE + row < B.height && block_col * BLOCK_SIZE + col < B.width)
	{
		return B.elements[(m * BLOCK_SIZE + row) * B.width + block_col * BLOCK_SIZE + col];
	}
	else
	{
		/* The elements of the tiles partially overlapping the matrice are set to zero (padding) */
		return 0.0;
	}
}

__device__ void SetElementMatrixC(Matrix C, int block_row, int block_col, int row, int col, float value) {

	if (block_row * BLOCK_SIZE + row < C.height && block_col * BLOCK_SIZE + col < C.width)
		C.elements[((block_row * BLOCK_SIZE + row) * C.width) + (block_col * BLOCK_SIZE) + col] = value;
}

// Matrix multiplication kernel  - SOLUTION
__global__ void MatMulKernelSolution(Matrix A, Matrix B, Matrix C) {

	// Shared memory used to store elements of A and B respectively 
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	// Block row and column 
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// Each thread computes one element of C by accumulating results into Cvalue 
	float Cvalue = 0.0;

	// Thread row and column within C 
	int row = threadIdx.y;
	int col = threadIdx.x;

	/*
		Loop over all the sub-matrices of A and B that are
		required to compute C
		Multiply each pair of sub-matrices together
		and accumulate the results
	*/
	for (int m = 0; m < (BLOCK_SIZE + A.width - 1) / BLOCK_SIZE; ++m) {

		// Each thread loads one element of each sub - matrix
		As[row][col] = GetElementMatrixA(A, m, blockRow, row, col);
		Bs[row][col] = GetElementMatrixB(B, m, blockCol, row, col);

		// Synchronize to make sure the sub-matrices are loaded before starting the computation 
		//__syncthreads();

		// Multiply A sub-matrix and B sub-matrix together 
		for (int e = 0; e < BLOCK_SIZE; ++e)
			Cvalue += As[row][e] * Bs[e][col];
		/*
		Synchronize to make sure that the preceding
		computation is done before loading two new
		sub-matrices of A and B in the next iteration
		*/
		//__syncthreads();
	}

	// Each thread block computes one sub-matrix of C 
	// Each thread writes one element 
	SetElementMatrixC(C, blockRow, blockCol, row, col, Cvalue);
}

void matMultiplication(const Matrix A, const Matrix B, const Matrix C) {
	// Load A to device memory 
	Matrix d_A;
	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	cudaError_t err = cudaMalloc(&d_A.elements, A.width * A.height * sizeof(float));
	cudaMemcpy(d_A.elements, A.elements, A.width * A.height * sizeof(float), cudaMemcpyHostToDevice);

	// Load B to device memory
	Matrix d_B;
	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	err = cudaMalloc(&d_B.elements, B.width * B.height * sizeof(float));
	cudaMemcpy(d_B.elements, B.elements, B.width * B.height * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate C in device memory 
	Matrix d_C;
	d_C.width = d_C.stride = C.width;
	d_C.height = C.height;
	err = cudaMalloc(&d_C.elements, C.width * C.height * sizeof(float));

	// Invoke kernel 
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	// Since the matrix dimensions might be not multiple of BLOCK_SIZE we have to take
	// as gridSize  the least integer that is greater than or equal to 
	// B.width / dimBlock.x for dimGrid.x and to
	// A.height / dimBlock.y for dimGrid.y
	dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);

	// NEW KERNEL
	MatMulKernelSolution << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

	//err = cudaThreadSynchronize();

	// Read C from device memory 
	err = cudaMemcpy(C.elements, d_C.elements, C.width * C.height * sizeof(float), cudaMemcpyDeviceToHost);

	// Free device memory 
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

struct Network {
	Matrix W1;
	Matrix W2;
	int noOfInputs;
	int noOfHidden;
	int noOfOutputs;

	Network(int inputs, int hidden, int outputs) {
		this->noOfInputs = inputs;
		this->noOfHidden = hidden;
		this->noOfOutputs = outputs;
	}
};

Matrix readInData(int height, int width) {
	Matrix input;
	input.height = 150;
	input.width = 5;
	input.elements = (float*)malloc(input.width * input.height * sizeof(float));

	FILE* fp;
	size_t count = 0;

	fp = fopen("iris.csv", "r");
	if (fp == NULL) {
		fprintf(stderr, "Error reading file\n");
	}

	while (fscanf(fp, "%f,%f,%f,%f,%f\n", &input.elements[count * input.width + 0], &input.elements[count * input.width + 1], &input.elements[count * input.width + 2], &input.elements[count * input.width + 3], &input.elements[count * input.width + 4]) == 5) {
		count++;
	}
	fclose(fp);

	return input;
}

Network generateWeights(Network network) {
	// Setting up weight matracies
	network.W1.height = network.noOfHidden;
	network.W1.width = network.noOfInputs;
	network.W1.elements = (float*)malloc(network.W1.width * network.W1.height * sizeof(float));
	network.W2.height = network.noOfOutputs;
	network.W2.width = network.noOfHidden;
	network.W2.elements = (float*)malloc(network.W2.width * network.W2.height * sizeof(float));

	// Setting weights to random values between 0 and 1 
	network.W1 = matRand(network.W1);
	network.W2 = matRand(network.W2);

	return network;
}

Matrix feedForward(Network network, Matrix input) {
	// Generating hidden ouputs
	Matrix net = matMult(network.W1, input);
	//Matrix hiddenOutputs = matAdd(hiddenInputs, network.B1);
	Matrix hiddenOutputs = activation(net);
	// Generating outputs
	Matrix outputs = matMult(network.W2, hiddenOutputs);
	//out = matAdd(out, network.B2);
	outputs = activation(outputs);
	return outputs;
}

Network train(Network network, Matrix input, Matrix target) {
	// FEEDFORWARD
	// Generating hidden ouputs
	Matrix net(network.W1.height, input.width);
	matMultiplication(network.W1, input, net);
	//Matrix net = matMult(network.W1, input);
	Matrix hiddenOutputs = activation(net);
	// Generating outputs
	Matrix outputs(network.W2.height, hiddenOutputs.width);
	matMultiplication(network.W2, hiddenOutputs, outputs);
	//Matrix outputs = matMult(network.W2, hiddenOutputs);
	outputs = activation(outputs);


	// START BACKPROP

	// Calculate output error
	Matrix outputError = matSub(target, outputs);

	// Calculate hidden error
	Matrix W2T = matTranspose(network.W2);
	Matrix hiddenError = matScale(W2T, outputError.elements[0]);

	// Calculate output gradients
	Matrix gradientOut = derivative(outputs);
	gradientOut = matElementMult(gradientOut, outputError);
	gradientOut = matScale(gradientOut, 0.1);

	//Calculate hidden gradients
	Matrix gradientHidden = derivative(hiddenOutputs);
	gradientHidden = matElementMult(gradientHidden, hiddenError);
	gradientHidden = matScale(gradientHidden, 0.1);

	Matrix hiddenOutputsT = matTranspose(hiddenOutputs);
	Matrix deltaOutput(gradientOut.height, hiddenOutputsT.width);
	matMultiplication(gradientOut, hiddenOutputsT, deltaOutput);
	//Matrix deltaOutput = matMult(gradientOut, hiddenOutputsT);
	network.W2 = matAdd(network.W2, deltaOutput);

	Matrix inputsT = matTranspose(input);
	Matrix deltaHidden(gradientHidden.height, inputsT.width);
	matMultiplication(gradientHidden, inputsT, deltaHidden);
	//Matrix deltaHidden = matMult(gradientHidden, inputsT);
	network.W1 = matAdd(network.W1, deltaHidden);

	free(net.elements); free(hiddenOutputs.elements); free(outputs.elements);
	free(outputError.elements); free(W2T.elements); free(hiddenError.elements);
	free(gradientOut.elements); free(gradientHidden.elements); free(hiddenOutputsT.elements);
	free(deltaOutput.elements); free(inputsT.elements); free(deltaHidden.elements);

	return network;
}

int main()
{
	srand(time(NULL));
	// Read in data from CSV file
	Matrix data = readInData(150, 5);
	// Define network parameters
	Network network(4, 40000, 1);
	// Generate intial weights for network
	network = generateWeights(network);

	Matrix input(4, 1);
	Matrix target(1, 1);

	for (int iterations = 0; iterations < 10; iterations++) {
		for (int i = 0; i < data.height; i++) {
			for (int j = 0; j < data.width - 1; j++) {
				input.elements[j] = data.elements[i * data.width + j];
			}
			target.elements[0] = data.elements[i * data.width + 4];
			network = train(network, input, target);
		}
	}

	input.elements[0] = 6.5;
	input.elements[1] = 2.8;
	input.elements[2] = 4.6;
	input.elements[3] = 1.5;

	Matrix output = feedForward(network, input);
}
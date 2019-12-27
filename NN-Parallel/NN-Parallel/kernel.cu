#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <cstddef>
#include <stddef.h>
#include "matrix.cu"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning(disable:4996)

// Thread block size 
#define BLOCK_SIZE 2

// Structure to define neural network
struct Network {
	Matrix W1; // First layer weights
	Matrix W2; // Second layer weights
	int noOfInputs; // Number of input nodes
	int noOfHidden; // Number of hidden nodes
	int noOfOutputs; // Number of output nodes
	// Constructor for neural network
	Network(int inputs, int hidden, int outputs) {
		this->noOfInputs = inputs;
		this->noOfHidden = hidden;
		this->noOfOutputs = outputs;
	}
};

// Function to get elements from Matrix A
__device__ float GetElementMatrixA(const Matrix A, int m, int block_row, int row, int col) {
	// Uses padding to 0 any elements that are overlapping in the matrix due to it not being a multiple of the tile size
	if ((m * BLOCK_SIZE + col < A.width) && (block_row * BLOCK_SIZE + row < A.height))
		return A.elements[(block_row * BLOCK_SIZE + row) * A.width + m * BLOCK_SIZE + col];
	else
		// Set elements to 0
		return 0.0;
}

// Function to get elemnts from Matrix B
__device__ float GetElementMatrixB(const Matrix B, int m, int block_col, int row, int col) {
	// Uses padding to 0 any elements that are overlapping in the matrix due to it not being a multiple of the tile size
	if (m * BLOCK_SIZE + row < B.height && block_col * BLOCK_SIZE + col < B.width)
		return B.elements[(m * BLOCK_SIZE + row) * B.width + block_col * BLOCK_SIZE + col];
	else
		// Set elements to 0
		return 0.0;
}

// Function to set the elements for the resulting matrix
__device__ void SetElementMatrixC(Matrix C, int block_row, int block_col, int row, int col, float value) {

	if (block_row * BLOCK_SIZE + row < C.height && block_col * BLOCK_SIZE + col < C.width)
		C.elements[((block_row * BLOCK_SIZE + row) * C.width) + (block_col * BLOCK_SIZE) + col] = value;
}

// Matrix multiplication kernel
__global__ void MatMulKernelSolution(Matrix A, Matrix B, Matrix C) {

	// Shared memory used to store elements of A and B
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// Each thread computes one element of C by accumulating results into Cvalue 
	float Cvalue = 0.0;
	int row = threadIdx.y;
	int col = threadIdx.x;

	//	Loop through each sub-matrix needed to calculate C and multiply each pair together and add results
	for (int m = 0; m < (BLOCK_SIZE + A.width - 1) / BLOCK_SIZE; ++m) {

		// Each thread loads one element of each sub-matrix
		As[row][col] = GetElementMatrixA(A, m, blockRow, row, col);
		Bs[row][col] = GetElementMatrixB(B, m, blockCol, row, col);

		// Sync threads to ensure the sub-matrices are loaded before starting the computation 
		__syncthreads();

		// Multiply A and B together 
		for (int j = 0; j < BLOCK_SIZE; ++j)
			Cvalue += As[row][j] * Bs[j][col];

		// Sync threads to ensure the final calculation is completed before loading two new matrices
		__syncthreads();
	}

	// Each thread block computes one sub-matrix of C and each thread writes one element
	SetElementMatrixC(C, blockRow, blockCol, row, col, Cvalue);
}

// Function to copy values to device and initiate kernel
void matMultiplication(const Matrix A, const Matrix B, const Matrix C) {
	// Load A to device memory 
	Matrix d_A;
	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

	// Load B to device memory
	Matrix d_B;
	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	err = cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

	// Allocate C in device memory 
	Matrix d_C;
	d_C.width = d_C.stride = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	err = cudaMalloc(&d_C.elements, size);

	// Invoke kernel 
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	// If matrix dimensions are not a multiple of the BLOCK_SIZE then we must use a ceiling function
	dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);

	// Call kerenel
	MatMulKernelSolution << <dimGrid, dimBlock >> > (d_A, d_B, d_C);
	err = cudaThreadSynchronize();

	// Read C from device memory 
	err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

	// Free device memory 
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// Function to read in data from a CSV file
Matrix readInData(int height, int width) {
	// Create matrix to store data
	Matrix input(150 , 5);
	// Loop through each line in file and store data is matrix
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
	// Return data from file
	return input;
}

// Function to randomly generate weights
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
	// Return whole network
	return network;
}

// Function to complete a feedforward pass
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

// Function to complete backpropagation algorithm and update weights
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

	// START BACKPROPAGATION
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

	// Calculate delta values for output layer and update weights
	Matrix hiddenOutputsT = matTranspose(hiddenOutputs);
	Matrix deltaOutput(gradientOut.height, hiddenOutputsT.width);
	matMultiplication(gradientOut, hiddenOutputsT, deltaOutput);
	network.W2 = matAdd(network.W2, deltaOutput);

	// Calculate delta values for hidden layer and update weights
	Matrix inputsT = matTranspose(input);
	Matrix deltaHidden(gradientHidden.height, inputsT.width);
	matMultiplication(gradientHidden, inputsT, deltaHidden);
	network.W1 = matAdd(network.W1, deltaHidden);

	// Clean up
	free(net.elements); free(hiddenOutputs.elements); free(outputs.elements);
	free(outputError.elements); free(W2T.elements); free(hiddenError.elements);
	free(gradientOut.elements); free(gradientHidden.elements); free(hiddenOutputsT.elements);
	free(deltaOutput.elements); free(inputsT.elements); free(deltaHidden.elements);
	// Return network with updated weights
	return network;
}

// Function to test the network and ouput the results
void testNetwork(Network network, Matrix data) {
	// Create input matrix
	Matrix input(4, 1);
	float error;

	// Loop through first 5 sets of data and run through a feedfoward pass to get the output
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < data.width - 1; j++) {
			input.elements[j] = data.elements[i * data.width + j]; // Store values in input matrix
			printf("Feature %d = %f | ", j + 1, input.elements[j]); // Print the features of the plant
		}
		Matrix output = feedForward(network, input); // Run through feedforward pass
		// Calculate error
		if (data.elements[i * data.width + 4] == 0)
			error = output.elements[0];
		else
			error = ((output.elements[0] - data.elements[i * data.width + 4]) / data.elements[i * data.width + 4]);
		// Print target, actual output and the error from the feed forward pass
		printf("Target = %f | Output  = %f | Error = %f\n", data.elements[i * data.width + 4], output.elements[0], error);
	}
}

int main()
{
	srand(time(NULL));
	// Read in data from CSV file
	Matrix data = readInData(150, 5);
	// Define network parameters
	Network network(4, 5, 1);
	// Generate intial weights for network
	network = generateWeights(network);
	// Create input and target matrices
	Matrix input(4, 1);
	Matrix target(1, 1);

	// Loop through each set of data and perform the backpropagation algorithm for a number of iterations
	for (int iterations = 0; iterations < 100; iterations++) {
		for (int i = 0; i < data.height; i++) {
			for (int j = 0; j < data.width - 1; j++) {
				input.elements[j] = data.elements[i * data.width + j]; // Store plant features in input matrix
			}
			target.elements[0] = data.elements[i * data.width + 4]; // Store target value in target matrix
			network = train(network, input, target); // Perform backpropagation and update weights
		}
	}
	// Call function to test the trained network
	testNetwork(network, data);
	cudaDeviceReset();
}
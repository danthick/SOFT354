#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
//#include "Matrix.cpp"

// Matrix structure. Matrices are stored in row-major order.
typedef struct {
	int width;
	int height;
	float* elements;
	int stride;
} Matrix;

Matrix matMult(const Matrix A, const Matrix B);
Matrix feedForward(Matrix W1, Matrix W2, Matrix input);
Matrix sigmoidFunction(Matrix input);
Matrix train(Matrix W1, Matrix W2, Matrix input);
Matrix matSub(const Matrix A, const Matrix B);


int main()
{
	srand(time(NULL));
	// Defining network variables
	int noOfInputs = 2; int noOfHiddenNodes = 3; int noOfOutputs = 1;

	// Defining weight matrices, accounting for bias on dimentions
	Matrix W1, W2;
	W1.height = noOfHiddenNodes;
	W1.width = noOfInputs + 1;
	W1.elements = (float*)malloc(W1.width * W1.height * sizeof(float));
	W2.height = noOfOutputs;
	W2.width = noOfHiddenNodes + 1;
	W2.elements = (float*)malloc(W2.width * W2.height * sizeof(float));

	// Setting weights to random values between 0 and 1 
	for (int i = 0; i < W1.height; i++)
		for (int j = 0; j < W1.width; j++)
			W1.elements[i * W1.width + j] = ((double)rand() / (RAND_MAX + 1.0));

	for (int i = 0; i < W2.height; i++)
		for (int j = 0; j < W2.width; j++)
			W2.elements[i * W2.width + j] = ((double)rand() / (RAND_MAX + 1.0));


	Matrix input;
	input.height = 2;
	input.width = 1;
	input.elements = (float*)malloc(input.width * input.height * sizeof(float));
	input.elements[0] = 2.5;
	input.elements[1] = 4.5;

	Matrix output = feedForward(W1, W2, input);

	for (int i = 0; i < output.height; i++)
		for (int j = 0; j < output.width; j++)
			printf("%f", output.elements[0]);

}

Matrix train(Matrix W1, Matrix W2, Matrix input, Matrix target) {
	// Add bias to the input matrix
	input.elements[input.height] = 1;
	// Calculate output from hidden layer
	Matrix net = matMult(W1, input);
	// Sigmoid activation function
	net = sigmoidFunction(net);
	// Add bias to activation from hidden layer, add update matrix dimensions
	net.elements[net.height] = 1;
	net.height++;
	// Calculate output from last layer
	net = matMult(W2, net);


	Matrix delta3 = matSub(target, net);
	for (int i = 0; i < delta3.height; i++) {
		for (int i = 0; i < delta3.width; i++) {
			delta3.elements[j + i * delta3.width] = -delta3.elements[j + i * delta3.width];
		}
	}



}



Matrix feedForward(Matrix W1, Matrix W2, Matrix input) {
	// Add bias to the input matrix
	input.elements[input.height] = 1;
	// Calculate output from hidden layer
	Matrix net = matMult(W1, input);
	// Sigmoid activation function
	net = sigmoidFunction(net);
	// Add bias to activation from hidden layer, add update matrix dimensions
	net.elements[net.height] = 1;
	net.height++;
	// Calculate output from last layer
	net = matMult(W2, net);
	return net;

}

Matrix sigmoidFunction(Matrix input) {
	for (int i = 0; i < input.height; i++) {
		for (int j = 0; j < input.width; j++) {
			input.elements[i * input.width + j] = 1 / (1+(exp(-input.elements[i * input.width + j])));
		}
	}
	return input;
}

// Function to complete a matrix mulitplication on the host
Matrix matMult(const Matrix A, const Matrix B) {
	Matrix C;
	C.height = A.height;
	C.width = B.width;
	C.elements = (float*)malloc(C.width * C.height * sizeof(float));

	for (int i = 0; i < A.height; i++) {
		for (int j = 0; j < B.width; j++) {
			C.elements[j + i * B.width] = 0;

			for (int k = 0; k < A.width; k++) {
				C.elements[j + i * B.width] += A.elements[k + i * A.width] * B.elements[j + k * B.width];
			}
		}
	}
	return C;
}

Matrix matSub(const Matrix A, const Matrix B) {
	Matrix C;
	C.height = A.height;
	C.width = A.width;
	C.elements = (float*)malloc(C.width * C.height * sizeof(float));

	for (int i = 0; i < C.height; i++) {
		for (int j = 0; j < C.width; j++) {
			C.elements[j + i * C.width] = A.elements[j + i * C.width] - B.elements[j + i * C.width];
		}
	}
	
}
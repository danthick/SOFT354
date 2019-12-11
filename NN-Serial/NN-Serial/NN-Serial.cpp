#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <vector>
#pragma warning(disable:4996)
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
std::vector<Matrix> train(Matrix W1, Matrix W2, Matrix input, Matrix target);
Matrix matSub(const Matrix A, const Matrix B);
Matrix matTranspose(const Matrix A);


int main()
{
	srand(time(NULL));
	// Defining network variables
	int noOfInputNodes = 2; int noOfHiddenNodes = 3; int noOfOutputs = 1;

	// Defining weight matrices, accounting for bias on dimentions
	Matrix W1, W2;
	W1.height = noOfHiddenNodes;
	W1.width = noOfInputNodes;
	//W1.width = noOfInputs + 1;
	W1.elements = (float*)malloc(W1.width * W1.height * sizeof(float));
	W2.height = noOfOutputs;
	W2.width = noOfHiddenNodes;
	//W2.width = noOfHiddenNodes + 1;
	W2.elements = (float*)malloc(W2.width * W2.height * sizeof(float));

	// Setting weights to random values between 0 and 1 
	for (int i = 0; i < W1.height; i++)
		for (int j = 0; j < W1.width; j++)
			W1.elements[i * W1.width + j] = ((double)rand() / (RAND_MAX + 1.0));

	for (int i = 0; i < W2.height; i++)
		for (int j = 0; j < W2.width; j++)
			W2.elements[i * W2.width + j] = ((double)rand() / (RAND_MAX + 1.0));


	float sepal_legnth = 0.0;
	float sepal_width = 0.0;
	float petal_legnth = 0.0;
	float petal_width = 0.0;
	float type = 0.0;

	FILE* fp = fopen("iris.csv", "r");

	int i = 0;
	int row = 150;
	float data[5][150];
	char line[150];
	while (fgets(line, 150, fp) && (i < row))
	{
		// double row[ssParams->nreal + 1];
		char* tmp = strdup(line);

		int j = 0;
		const char* tok;
		for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, "\n"))
		{
			data[i][j] = atof(tok);
			printf("%f", data[i][j]);
		}
		printf("\n");

		free(tmp);
		i++;
	}


	Matrix input;
	input.height = 2;
	input.width = 1;
	input.elements = (float*)malloc(input.width * input.height * sizeof(float));
	input.elements[0] = 2.5;
	input.elements[1] = 4.5;
	Matrix target;
	target.height = 2;
	target.width = 1;
	target.elements = (float*)malloc(target.width * target.height * sizeof(float));
	target.elements[0] = 20.2345678;
	target.elements[1] = 9;

	
	std::vector<Matrix> weights = train(W1, W2, input, input);
	for (int i = 0; i < 5000; i++) {
		weights = train(weights[0], weights[1], input, target);
	}

	Matrix output = feedForward(weights[0], weights[1], input);

}

std::vector<Matrix> train(Matrix W1, Matrix W2, Matrix input, Matrix target) {
	// Add bias to the input matrix
	input.elements[input.height] = 1;
	// Calculate output from hidden layer
	Matrix net = matMult(W1, input);
	// Sigmoid activation function
	Matrix a2 = sigmoidFunction(net);
	// Add bias to activation from hidden layer, and update matrix dimensions
	Matrix a2Hat = a2;
	a2Hat.elements[net.height] = 1;
	a2Hat.height++;
	// Calculate output from last layer
	Matrix o = matMult(W2, net);


	// Performing matrix subtraction
	Matrix delta3 = matSub(target, o);
	// Changing all values to negative
	for (int i = 0; i < delta3.height; i++) {
		for (int j = 0; j < delta3.width; j++) {
			delta3.elements[j + i * delta3.width] = -delta3.elements[j + i * delta3.width];
		}
	}
	

	// NEED TO REMOVE BIAS FROM W2
	for (int i = 0; i < W2.height; i++) {
		for (int j = 1; j < W2.width; j++) {
			if (j % (W2.width - 1) == 0) {
				//printf("4th element will be removed\n\n");
			}
		}
	}

	// HAVE NOT INCLUDED ACTIVATION FUNCTION REMOVAL
	Matrix W2T = matTranspose(W2);
	Matrix delta2 = matMult(W2T, delta3);

	// Calculating error gradient
	Matrix inputT = matTranspose(input);
	Matrix netT = matTranspose(net);

	Matrix errGradientW1 = matMult(delta2, inputT);
	Matrix errGradientW2 = matMult(delta3, netT);

	// Updating weights using learning rate
	errGradientW1.elements[0] = errGradientW1.elements[0] * 0.01;
	errGradientW2.elements[0] = errGradientW2.elements[0] * 0.01;

	W1 = matSub(W1, errGradientW1);
	W2 = matSub(W2, errGradientW2);


	std::vector<Matrix> weights;
	weights.push_back(W1);
	weights.push_back(W2);
	
	return weights;

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
	return C;
}

Matrix matTranspose(const Matrix A) {
	Matrix C;
	C.height = A.width;
	C.width = A.height;
	C.elements = (float*)malloc(C.width * C.height * sizeof(float));

	for (int row = 0; row < A.height; row++)
		for (int col = 0; col < A.width; col++)
			C.elements[row + col * A.height] = A.elements[col + row * A.width];

	return C;
}
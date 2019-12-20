#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <cstddef>
#include <stddef.h>
#include "Matrix.cpp"
#pragma warning(disable:4996)

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
	Matrix net = matMult(network.W1, input);
	Matrix hiddenOutputs = activation(net);
	// Generating outputs
	Matrix outputs = matMult(network.W2, hiddenOutputs);
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
	Matrix deltaOutput = matMult(gradientOut, hiddenOutputsT);
	network.W2 = matAdd(network.W2, deltaOutput);

	Matrix inputsT = matTranspose(input);
	Matrix deltaHidden = matMult(gradientHidden, inputsT);
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
	Network network(4,5,1);
	// Generate intial weights for network
	network = generateWeights(network);

	Matrix input(4, 1);
	Matrix target(1, 1);

	for (int iterations = 0; iterations < 5000; iterations++) {
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
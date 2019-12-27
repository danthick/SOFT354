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

// Function to read in data from a CSV file
Matrix readInData(int height, int width) {
	// Create matrix to store data
	Matrix input(150, 5);
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
	Matrix net = matMult(network.W1, input);
	Matrix hiddenOutputs = activation(net);
	// Generating outputs
	Matrix outputs = matMult(network.W2, hiddenOutputs);
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
	Matrix deltaOutput = matMult(gradientOut, hiddenOutputsT);
	network.W2 = matAdd(network.W2, deltaOutput);

	// Calculate delta values for hidden layer and update weights
	Matrix inputsT = matTranspose(input);
	Matrix deltaHidden = matMult(gradientHidden, inputsT);
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
	Network network(4,5,1);
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
}
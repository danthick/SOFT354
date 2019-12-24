#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <cstddef>
#include <stddef.h>

/*
Small matrix library to perform various required matrix calculations.
*/

// Structure to define the matrix features. Elements stored in row major order.
struct Matrix {
	int height;
	int width;
	float* elements;
	int stride;

	Matrix() {}

	// Constructor for a matrix stucture
	Matrix(int height, int width) {
		this->height = height;
		this->width = width;
		this->elements = (float*)malloc(height * width * sizeof(float));
	}
};

// Function to fill a matrix with random values between 0 and 1
static Matrix matRand(Matrix A) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			C.elements[i * C.width + j] = ((double)rand() / (RAND_MAX + 1.0));
	return C;
}

// Function to multiply two matrices together
static Matrix matMult(const Matrix A, const Matrix B) {
	Matrix C(A.height, B.width);
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

// Function to subtract one matrix from another element wise
static Matrix matSub(const Matrix A, const Matrix B) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < C.height; i++) {
		for (int j = 0; j < C.width; j++) {
			C.elements[j + i * C.width] = A.elements[j + i * C.width] - B.elements[j + i * C.width];
		}
	}
	return C;
}

// Function to transpose a matrix
static Matrix matTranspose(const Matrix A) {
	Matrix C(A.width, A.height);
	for (int row = 0; row < A.height; row++)
		for (int col = 0; col < A.width; col++)
			C.elements[row + col * A.height] = A.elements[col + row * A.width];

	return C;
}

// Function to add two matrices together alement wise
static Matrix matAdd(const Matrix A, const Matrix B) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			C.elements[j + i * C.width] = A.elements[j + i * C.width] + B.elements[j + i * C.width];
	return C;
}

// Function to multiply each matrix element by a given value
static Matrix matScale(const Matrix A, float B) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			C.elements[j + i * C.width] = A.elements[j + i * A.width] * B;
	return C;
}

// Function to multiply two matrices together element wise
static Matrix matElementMult(const Matrix A, const Matrix B) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			C.elements[j + i * C.width] = A.elements[j + i * A.width] * B.elements[j + i * B.width];
	return C;
}

// Function to run sigmoid activation to each element in the matrix
static Matrix activation(const Matrix A) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			C.elements[j + i * A.width] = 1 / (1 + exp(-A.elements[j + i * A.width]));
	return C;
}

// Function to calculate the sigmoid derivative for each element in the matrix
static Matrix derivative(const Matrix A) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			C.elements[j + i * A.width] = A.elements[j + i * A.width] * (1 - A.elements[j + i * A.width]);
	return C;
}
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

struct Matrix {
	int height;
	int width;
	float* elements;
	int stride;

	Matrix(){}

	Matrix(int height, int width) {
		this->height = height;
		this->width = width;
		this->elements = (float*)malloc(height * width * sizeof(float));
	}
};


static Matrix matRand(Matrix A) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			C.elements[i * C.width + j] = ((double)rand() / (RAND_MAX + 1.0));
	return C;
}

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

static Matrix matSub(const Matrix A, const Matrix B) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < C.height; i++) {
		for (int j = 0; j < C.width; j++) {
			C.elements[j + i * C.width] = A.elements[j + i * C.width] - B.elements[j + i * C.width];
		}
	}
	return C;
}

static Matrix matTranspose(const Matrix A) {
	Matrix C(A.width, A.height);
	for (int row = 0; row < A.height; row++)
		for (int col = 0; col < A.width; col++)
			C.elements[row + col * A.height] = A.elements[col + row * A.width];

	return C;
}

static Matrix matAdd(const Matrix A, const Matrix B) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			C.elements[j + i * C.width] = A.elements[j + i * C.width] + B.elements[j + i * C.width];
	return C;
}

static Matrix matScale(const Matrix A, float B) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			C.elements[j + i * C.width] = A.elements[j + i * A.width] * B;
	return C;
}

static Matrix matElementMult(const Matrix A, const Matrix B) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			C.elements[j + i * C.width] = A.elements[j + i * A.width] * B.elements[j + i * B.width];
	return C;
}


static Matrix activation(const Matrix A) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			C.elements[j + i * A.width] = 1 / (1 + exp(-A.elements[j + i * A.width]));
	return C;
}

static Matrix derivative(const Matrix A) {
	Matrix C(A.height, A.width);
	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			C.elements[j + i * A.width] = A.elements[j + i * A.width] * (1 - A.elements[j + i * A.width]);
	return C;
}
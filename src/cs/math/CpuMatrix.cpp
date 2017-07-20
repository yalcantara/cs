/*
 * CpuMatrix.cpp
 *
 *  Created on: Feb 5, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/core/Exception.h>
#include <cs/core/lang.h>
#include <cs/math/CpuMatrix.h>
#include <cs/math/math.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cs {
using namespace core;
namespace math {

CpuMatrix::CpuMatrix(size_t m, size_t n) :
		CpuMatrix(m, n, true) {
}

CpuMatrix::CpuMatrix(size_t m, size_t n, bool clear) :
		Matrix(m, n) {
	
	if (clear) {
		arr = (float*) calloc(length, sizeof(float));
	} else {
		arr = (float*) malloc(sizeof(float) * length);
	}
}

CpuMatrix::CpuMatrix(size_t m, size_t n, float* src) :
		CpuMatrix(m, n, false) {
	copy_float(src, arr, length);
}

CpuMatrix::CpuMatrix(const CpuMatrix& other) :
		CpuMatrix(other.m, other.n, false) {
	copy_float(other.arr, arr, length);
}

CpuMatrix::CpuMatrix(const initializer_list<const initializer_list<float>> &list) :
		Matrix(1, 1) {
	size_t listSize = list.size();
	if (listSize < 1) {
		throw Exception("Invalid list size: " + to_string(listSize) + ".");
	}
	
	const initializer_list<float>* start = list.begin();
	
	size_t listColumns = start->size();
	if (listSize < 1) {
		throw Exception("Invalid list columns: " + to_string(listColumns) + ".");
	}
	//first let's check that each row has the same number of columns
	for (size_t i = 0; i < listSize; i++) {
		const initializer_list<float>& crt = start[i];
		size_t crtColumns = crt.size();
		
		if (listColumns != crtColumns) {
			throw Exception(
					"The number of columns does not match as the first row. First row columns: "
							+ to_string(listColumns) + ", row " + to_string(i) + " columns: " + to_string(crtColumns)
							+ ".");
		}
	}
	
	const_cast<size_t&>(m) = listSize;
	const_cast<size_t&>(n) = listColumns;
	const_cast<size_t&>(length) = m * n;
	arr = (float*) malloc(sizeof(float) * length);
	
	for (size_t i = 0; i < listSize; i++) {
		const initializer_list<float>& crt = start[i];
		const float* rowStart = crt.begin();
		
		for (size_t j = 0; j < listColumns; j++) {
			arr[i * n + j] = rowStart[j];
		}
	}
}

void CpuMatrix::randn() {
	cs::math::randn(arr, length);
}

void CpuMatrix::clear() {
	for (size_t i = 0; i < length; i++) {
		arr[i] = 0.0;
	}
}

CpuMatrix& CpuMatrix::operator=(const CpuMatrix& other) {
	if (&other == this) {
		return *this;
	}
	//There are 2 cases:
	//a - the destination is not initialized
	//b - the dimensions are equals (if not, throw an exception)
	
	if (arr == nullptr) {
		
		//allocate and copy
		const_cast<size_t&>(m) = other.m;
		const_cast<size_t&>(n) = other.n;
		const_cast<size_t&>(length) = other.length;
		arr = (float*) malloc(sizeof(float) * length);
		copy_float(other.arr, arr, length);
		
	} else {
		check_same_dimensions(other);
		//the vector is already initialized
		//just copy
		copy_float(other.arr, arr, length);
		
	}
	
	return *this;
}

float CpuMatrix::at(size_t idx) const {
	check_index(idx);
	return arr[idx];
}

float CpuMatrix::get(size_t i, size_t j) const {
	check_index(i, j);
	return arr[i * n + j];
}

void CpuMatrix::set(size_t i, size_t j, float val) const {
	check_index(i, j);
	arr[i * n + j] = val;
}

void CpuMatrix::subi(const CpuMatrix& b) {
	check_same_dimensions(b);
	
	size_t l = length;
	
	float* A = arr;
	float* B = b.arr;
	
	for (size_t i = 0; i < l; i++) {
		A[i] = A[i] - B[i];
	}
}

const CpuMatrix CpuMatrix::operator+(const CpuMatrix& b) const {
	check_same_dimensions(b);
	
	size_t l = length;
	CpuMatrix c = CpuMatrix(m, n, false);
	
	float* A = arr;
	float* B = b.arr;
	float* C = c.arr;
	
	for (size_t i = 0; i < l; i++) {
		C[i] = A[i] + B[i];
	}
	
	return c;
}

const CpuMatrix CpuMatrix::operator+(float val) const {
	
	size_t l = length;
	CpuMatrix ans = CpuMatrix(m, n, false);
	
	float* A = arr;
	float* B = ans.arr;
	
	for (size_t i = 0; i < l; i++) {
		B[i] = A[i] + val;
	}
	
	return ans;
}

const CpuMatrix CpuMatrix::operator-(const CpuMatrix& b) const {
	check_same_dimensions(b);
	
	size_t l = length;
	CpuMatrix c = CpuMatrix(m, n, false);
	
	float* A = arr;
	float* B = b.arr;
	float* C = c.arr;
	
	for (size_t i = 0; i < l; i++) {
		C[i] = A[i] - B[i];
	}
	
	return c;
}

const CpuMatrix CpuMatrix::operator-(float val) const {
	
	size_t l = length;
	CpuMatrix ans = CpuMatrix(m, n, false);
	
	float* A = arr;
	float* B = ans.arr;
	
	for (size_t i = 0; i < l; i++) {
		B[i] = A[i] - val;
	}
	
	return ans;
}

const CpuMatrix CpuMatrix::operator-() const {
	
	size_t l = length;
	CpuMatrix ans = CpuMatrix(m, n, false);
	
	float* A = arr;
	float* B = ans.arr;
	
	for (size_t i = 0; i < l; i++) {
		B[i] = -A[i];
	}
	
	return ans;
}

const CpuMatrix CpuMatrix::operator*(const CpuMatrix& b) const {
	check_same_dimensions(b);
	
	size_t l = length;
	CpuMatrix c = CpuMatrix(m, n, false);
	
	float* A = arr;
	float* B = b.arr;
	float* C = c.arr;
	
	for (size_t i = 0; i < l; i++) {
		C[i] = A[i] * B[i];
	}
	
	return c;
}

const CpuMatrix CpuMatrix::operator*(float scalar) const {
	
	size_t l = length;
	CpuMatrix ans = CpuMatrix(m, n, false);
	
	float* A = arr;
	float* B = ans.arr;
	
	for (size_t i = 0; i < l; i++) {
		B[i] = A[i] * scalar;
	}
	
	return ans;
}

const CpuMatrix CpuMatrix::operator/(const CpuMatrix& b) const {
	check_same_dimensions(b);
	
	size_t l = length;
	CpuMatrix c = CpuMatrix(m, n, false);
	
	float* A = arr;
	float* B = b.arr;
	float* C = c.arr;
	
	for (size_t i = 0; i < l; i++) {
		C[i] = A[i] / B[i];
	}
	
	return c;
}

const CpuMatrix CpuMatrix::operator/(float scalar) const {
	
	size_t l = length;
	CpuMatrix ans = CpuMatrix(m, n, false);
	
	float* A = arr;
	float* B = ans.arr;
	
	for (size_t i = 0; i < l; i++) {
		B[i] = A[i] / scalar;
	}
	
	return ans;
}

const CpuMatrix CpuMatrix::operator^(float exp) const {
	
	size_t l = length;
	CpuMatrix ans = CpuMatrix(m, n, false);
	
	float* A = arr;
	float* B = ans.arr;
	
	for (size_t i = 0; i < l; i++) {
		B[i] = pow(A[i], exp);
	}
	
	return ans;
}

void CpuMatrix::dot(const CpuMatrix& b, CpuMatrix& ans) const {
	
	assert_cols(b.m, n);
	
	assert_rows(ans.m, m);
	assert_cols(ans.n, b.n);
	
	const size_t m = this->m;
	const size_t n = this->n;
	const size_t p = b.n;
	
	float* A = arr;
	float* B = b.arr;
	float* C = ans.arr;
	
	ans.clear();
	//algorithm based on the GNU Scientific Library (GSL)
	//linear algebra method: gsl_blas_sgemm
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			const float pivot = A[i * n + j];
			for (size_t k = 0; k < p; k++) {
				C[i * p + k] += pivot * B[j * p + k];
			}
		}
	}
}

const CpuMatrix CpuMatrix::dot(const CpuMatrix& b) const {
	
	CpuMatrix ans = CpuMatrix(m, b.n);
	
	dot(b, ans);
	
	return ans;
}

const CpuVector CpuMatrix::dot(const CpuVector& b) const {
	
	assert_cols(b.length, n);
	
	CpuVector ans = CpuVector(m, false);
	
	const size_t m = this->m;
	const size_t n = this->n;
	
	float* A = arr;
	float* B = b.ptr();
	float* C = ans.ptr();
	
	for (size_t i = 0; i < m; i++) {
		float val = 0.0;
		for (size_t j = 0; j < n; j++) {
			val += A[i * n + j] * B[j];
		}
		C[i] = val;
	}
	
	return ans;
}

const CpuMatrix CpuMatrix::affine(const CpuMatrix& x, const CpuVector& b) const {
	
	CpuMatrix ans = CpuMatrix(m, x.n, false);
	affine(x, b, ans);
	
	return ans;
}

void CpuMatrix::affine(const Matrix& x, const Vector& b, Matrix& ans) const {
	affine(cpu_cast(x), cpu_cast(b), cpu_cast(ans));
}

void CpuMatrix::affine(const CpuMatrix& x, const CpuVector& b, CpuMatrix& ans) const {
	
	size_t p = x.n;
	assert_rows(b.length, p);
	dot(x, ans);
	
	float* B = b.ptr();
	float* Y = ans.ptr();
	
	for (size_t i = 0; i < m; i++) {
		for (size_t k = 0; k < p; k++) {
			Y[i * p + k] += B[k];
		}
	}
}

float CpuMatrix::sum() const {
	
	size_t l = length;
	float* A = arr;
	
	float ans = A[0];
	for (size_t i = 1; i < l; i++) {
		ans += A[i];
	}
	
	return ans;
}

float CpuMatrix::max() const {
	
	size_t l = length;
	float* A = arr;
	
	float ans = A[0];
	for (size_t i = 1; i < l; i++) {
		ans = std::max(ans, A[i]);
	}
	
	return ans;
}

float CpuMatrix::min() const {
	
	size_t l = length;
	float* A = arr;
	
	float ans = A[0];
	for (size_t i = 1; i < l; i++) {
		ans = std::min(ans, A[i]);
	}
	
	return ans;
}

float CpuMatrix::avg() const {
	return sum() / length;
}

void CpuMatrix::copy(Matrix& dest) const {
	copy(cpu_cast(dest));
}

void CpuMatrix::copy(CpuMatrix& dest) const {
	check_same_dimensions(dest);
	copy_float(arr, dest.arr, length);
}

const CpuMatrix CpuMatrix::sltcols(size_t start, size_t end) const {
	
	if (start >= end) {
		throw Exception(
				"The start parameter must be less than the end parameter -1. The start parameter is: "
						+ to_string(start) + ", the end parameter is: " + to_string(end) + ".");
	}
	
	if (start >= n - 1) {
		throw Exception(
				"The start parameter must be less than the number of columns -1. Expected < " + to_string(n - 1)
						+ ", but got: " + to_string(start) + " instead.");
	}
	
	if (end > n) {
		throw Exception(
				"The end parameter must be less than the number of columns. Expected < " + to_string(n) + ", but got: "
						+ to_string(end) + " instead.");
	}
	
	size_t cols = end - start;
	CpuMatrix ans = CpuMatrix(m, cols);
	
	float* A = this->arr;
	float* B = ans.ptr();
	
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < cols; j++) {
			size_t src = i * n + j + start;
			size_t dst = i * cols + j;
			
			B[dst] = A[src];
		}
	}
	
	return ans;
}

float* CpuMatrix::ptr() const {
	return arr;
}

void CpuMatrix::print() const {
	
	size_t rows = std::min((size_t) MATRIX_PRINT_MAX, m);
	size_t cols = std::min((size_t) MATRIX_PRINT_MAX, n);
	
	if (m > MATRIX_PRINT_MAX || cols > MATRIX_PRINT_MAX) {
		printf("Matrix  %dx%d   (truncated)\n", (int) m, (int) n);
	} else {
		printf("Matrix  %dx%d\n", (int) m, (int) n);
	}
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			printf("%12.4f", arr[i * n + j]);
			if (j + 1 < n) {
				printf("  ");
			}
		}
		println();
	}
	
	println();
}

CpuMatrix::~CpuMatrix() {
	if (arr) {
		free(arr);
	}
}

} // namespace math 
} // namespace cs 

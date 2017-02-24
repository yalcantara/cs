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

CpuMatrix& CpuMatrix::cpu_cast(Matrix& m) const {
	return cpu_cast(&m);
}

CpuMatrix& CpuMatrix::cpu_cast(Matrix* m) const {
	CpuMatrix* ans = dynamic_cast<CpuMatrix*>(m);
	if (ans) {
		return *ans;
	}
	throw new Exception("Class cast exception. The pointer could not be casted into a CpuMatrix.");
}

CpuVector& CpuMatrix::cpu_cast(Vector* m) const {
	CpuVector* ans = dynamic_cast<CpuVector*>(m);
	if (ans) {
		return *ans;
	}
	throw new Exception("Class cast exception. The pointer could not be casted into a CpuVector.");
}

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
			throw new Exception(
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

void CpuMatrix::dot(CpuMatrix& b, CpuMatrix& ans) {
	
	assert_cols(b.m, n);
	
	assert_rows(ans.m, m);
	assert_cols(ans.n, b.n);
	
	const size_t m = this->m;
	const size_t n = this->n;
	const size_t p = b.n;
	
	float* A = arr;
	float* B = b.arr;
	float* C = ans.arr;
	
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
	CpuMatrix& self = const_cast<CpuMatrix&>(*this);
	CpuMatrix& cb = const_cast<CpuMatrix&>(b);
	
	self.dot(cb, ans);
	
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
	
	assert_cols(b.length, x.n);
	const CpuMatrix ans = dot(x);
	
	const size_t m = ans.m;
	const size_t p = ans.n;
	
	float* B = b.ptr();
	float* Y = ans.ptr();
	for (size_t i = 0; i < m; i++) {
		for (size_t k = 0; k < p; k++) {
			Y[i * p + k] += B[k];
		}
	}
	
	return ans;
}

void CpuMatrix::affine(Matrix* x, Vector* b, Matrix* ans) {
	
	CpuMatrix& cx = cpu_cast(x);
	CpuVector& cb = cpu_cast(b);
	CpuMatrix& cans = cpu_cast(ans);
	
	assert_rows(cb.length, cx.n);
	assert_rows(cans.m, cx.m);
	dot(cx, cans);
	
	float* A = cans.arr;
	float* B = cb.ptr();
	
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			arr[i * n + j] += B[j];
		}
	}
}

void CpuMatrix::randn() {
	cs::math::randn(arr, length);
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

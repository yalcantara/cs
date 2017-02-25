/*
 * GpuMatrix.cpp
 *
 *  Created on: Feb 9, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/core/Exception.h>
#include <cs/gpu/gpu.h>
#include <cs/math/GpuMatrix.h>
#include <cs/math/math.h>
#include <stdlib.h>

namespace cs {
using namespace core;
using namespace gpu;
namespace math {

GpuMatrix::GpuMatrix(size_t m, size_t n) :
		GpuMatrix(m, n, true) {
}

GpuMatrix::GpuMatrix(size_t m, size_t n, bool clear) :
		Matrix(m, n) {
	devPtr = gpu_malloc(length, clear);
}

GpuMatrix::GpuMatrix(const CpuMatrix& other) :
		GpuMatrix(other.m, other.n, false) {
	float* src = other.ptr();
	copy_cpu_to_gpu(src, devPtr, length);
}

GpuMatrix::GpuMatrix(const initializer_list<const initializer_list<float>> &list) :
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
	float* temp = (float*) malloc(sizeof(float) * length);
	
	for (size_t i = 0; i < listSize; i++) {
		const initializer_list<float>& crt = start[i];
		const float* rowStart = crt.begin();
		
		for (size_t j = 0; j < listColumns; j++) {
			temp[i * n + j] = rowStart[j];
		}
	}
	
	devPtr = gpu_malloc(length, false);
	copy_cpu_to_gpu(temp, devPtr, length);
	free(temp);
}

void GpuMatrix::clear() {
	gpu_set(devPtr, 0, length);
}

void GpuMatrix::randn() {
	CpuMatrix c = cs::math::randn(m, n);
	
	float* src = c.ptr();
	copy_cpu_to_gpu(src, devPtr, length);
}

GpuMatrix& GpuMatrix::operator=(const GpuMatrix& other) {
	
	if (&other == this) {
		return *this;
	}
	
	//There are 2 cases:
	//a - the destination is not initialized
	//b - the dimensions are equals (if not, throw an exception)
	
	if (devPtr == nullptr) {
		
		//allocate and copy
		const_cast<size_t&>(length) = other.length;
		devPtr = gpu_malloc(other.length, false);
		copy_gpu_to_gpu(other.devPtr, devPtr, length);
	} else {
		check_same_dimensions(other);
		//the vector is already initialized
		//just copy
		copy_gpu_to_gpu(other.devPtr, devPtr, length);
	}
	
	return *this;
}

const GpuMatrix GpuMatrix::operator+(const GpuMatrix& b) const {
	check_same_dimensions(b);
	
	GpuMatrix ans = GpuMatrix(m, n, false);
	
	gpu_add(devPtr, b.devPtr, ans.devPtr, length);
	
	return ans;
}

const GpuMatrix GpuMatrix::operator-() const {
	return (*this) * -1;
}

const GpuMatrix GpuMatrix::operator*(const GpuMatrix& b) const {
	check_same_dimensions(b);
	GpuMatrix ans = GpuMatrix(m, n, false);
	
	gpu_mult(devPtr, b.devPtr, ans.devPtr, length);
	
	return ans;
}

const GpuMatrix GpuMatrix::operator*(const float scalar) const {
	
	GpuMatrix ans = GpuMatrix(m, n, false);
	
	gpu_mult(devPtr, scalar, ans.devPtr, length);
	
	return ans;
}

const GpuMatrix GpuMatrix::operator/(const GpuMatrix& b) const {
	check_same_dimensions(b);
	GpuMatrix ans = GpuMatrix(m, n, false);
	
	gpu_div(devPtr, b.devPtr, ans.devPtr, length);
	
	return ans;
}

const GpuMatrix GpuMatrix::operator/(const float scalar) const {
	
	GpuMatrix ans = GpuMatrix(m, n, false);
	float div = 1.0 / scalar;
	gpu_mult(devPtr, div, ans.devPtr, length);
	
	return ans;
}

const GpuMatrix GpuMatrix::operator^(const float exp) const {
	
	GpuMatrix ans = GpuMatrix(m, n, false);
	
	gpu_pow(devPtr, exp, ans.devPtr, length);
	
	return ans;
}

void GpuMatrix::addi(const GpuMatrix& b) {
	check_same_dimensions(b);
	gpu_add_inplace(devPtr, b.devPtr, length);
}
void GpuMatrix::subi(const GpuMatrix& b) {
	check_same_dimensions(b);
	gpu_sub_inplace(devPtr, b.devPtr, length);
}
void GpuMatrix::multi(const GpuMatrix& b) {
	check_same_dimensions(b);
	gpu_mult_inplace(devPtr, b.devPtr, length);
}
void GpuMatrix::multi(const float scalar) {
	gpu_mult_inplace(devPtr, scalar, length);
}
void GpuMatrix::divi(const GpuMatrix& b) {
	check_same_dimensions(b);
	gpu_div_inplace(devPtr, b.devPtr, length);
}
void GpuMatrix::divi(const float scalar) {
	//inverted scalar
	const float div = 1.0 / scalar;
	gpu_mult_inplace(devPtr, div, length);
}
void GpuMatrix::powi(const float exp) {
	gpu_pow_inplace(devPtr, exp, length);
}

const GpuMatrix GpuMatrix::dot(const GpuMatrix& b) const {
	
	assert_rows(b.m, n);
	
	const size_t p = b.n;
	GpuMatrix ans = GpuMatrix(m, p);
	
	gpu_dot(devPtr, b.devPtr, ans.devPtr, m, n, p);
	
	return ans;
}

void GpuMatrix::dot(GpuMatrix& b, GpuMatrix& ans) {
	
	assert_rows(b.m, n);
	
	assert_rows(ans.m, m);
	assert_cols(ans.n, b.n);
	
	gpu_dot(devPtr, b.devPtr, ans.devPtr, m, n, b.n);
}

const GpuVector GpuMatrix::dot(const GpuVector& b) const {
	
	assert_rows(b.length, n);
	
	GpuVector ans = GpuVector(m);
	float* B = b.ptr();
	float* C = ans.ptr();
	gpu_dot(devPtr, B, C, m, n);
	
	return ans;
}

const GpuMatrix GpuMatrix::affine(const GpuMatrix& x, const GpuVector& b) const {
	
	assert_rows(b.length, x.n);
	GpuMatrix ans = dot(x);
	
	float* A = ans.devPtr;
	float* B = b.ptr();
	
	gpu_broadcast_sum_rows(A, B, A, ans.m, ans.n);
	
	return ans;
}

void GpuMatrix::affine(Matrix* x, Vector* b, Matrix* ans) {
	
	GpuMatrix& gx = gpu_cast(x);
	GpuVector& gb = gpu_cast(b);
	GpuMatrix& gans = gpu_cast(ans);
	
	assert_rows(gb.length, gx.n);
	assert_rows(gans.m, gx.m);
	dot(gx, gans);
	
	float* A = gans.devPtr;
	float* B = gb.ptr();
	
	gpu_broadcast_sum_rows(A, B, A, gans.m, gans.n);
}

float GpuMatrix::sum() const {
	
	float ans = gpu_sum(devPtr, length);
	
	return ans;
}

const CpuMatrix GpuMatrix::cpu() const {
	
	CpuMatrix ans = CpuMatrix(m, n, false);
	
	float* dest = ans.ptr();
	copy_gpu_to_cpu(devPtr, dest, length);
	
	return ans;
}

void GpuMatrix::print() const {
	CpuMatrix a = cpu();
	a.print();
}

GpuMatrix::~GpuMatrix() {
	if (devPtr) {
		gpu_free(devPtr);
	}
}

} // namespace math
} // namespace cs

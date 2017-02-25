/*
 * CpuVector.cpp
 *
 *  Created on: Jan 28, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/core/Exception.h>
#include <cs/core/lang.h>
#include <cs/math/CpuVector.h>
#include <cs/math/math.h>
#include <stddef.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cs {
using namespace core;
namespace math {

CpuVector::CpuVector(size_t length) :
		CpuVector(length, true) {
	
}

CpuVector::CpuVector(size_t length, bool clear) :
		length(length) {
	
	if (length < 1) {
		throw Exception("Invalid length " + to_string(length) + ".");
	}
	
	if (clear) {
		arr = (float*) calloc(length, sizeof(float));
	} else {
		arr = (float*) malloc(sizeof(float) * length);
	}
}

CpuVector::CpuVector(const CpuVector& other) :
		CpuVector(other.length, false) {
	float* src = other.ptr();
	copy_float(src, arr, length);
}

CpuVector::CpuVector(const initializer_list<float> &list) :
		CpuVector(list.size(), false) {
	const float* start = list.begin();
	
	for (size_t i = 0; i < length; i++) {
		arr[i] = *(start + i);
	}
}

void CpuVector::check_index(size_t idx) const {
	if (idx >= length) {
		throw Exception(
				"Index out of bounds. Expected < " + to_string(length) + ", but got: " + to_string(idx) + " instead.");
	}
}

void CpuVector::check_same_length(const CpuVector& other) const {
	if (other.length != length) {
		throw Exception(
				"The length must be the same. Expected " + to_string(length) + ", but got: " + to_string(other.length)
						+ " instead.");
	}
}

void CpuVector::randn() {
	cs::math::randn(arr, length);
}

void CpuVector::clear() {
	for (size_t i = 0; i < length; i++) {
		arr[i] = 0.0;
	}
}

CpuVector& CpuVector::operator=(const CpuVector& other) {
	if (&other == this) {
		return *this;
	}
	//There are 2 cases:
	//a - the destination is not initialized
	//b - the dimensions are equals (if not, throw an exception)
	
	if (arr == nullptr) {
		
		//allocate and copy
		const_cast<size_t&>(length) = other.length;
		arr = (float*) malloc(sizeof(float) * length);
		copy_float(other.arr, arr, length);
		
	} else {
		check_same_length(other);
		//the vector is already initialized
		//just copy
		copy_float(other.arr, arr, length);
		
	}
	
	return *this;
}

float CpuVector::operator[](size_t idx) const {
	check_index(idx);
	return arr[idx];
}

float& CpuVector::operator[](size_t idx) {
	check_index(idx);
	return arr[idx];
}

const CpuVector CpuVector::operator+(const CpuVector& b) const {
	check_same_length(b);
	
	size_t l = length;
	CpuVector c = CpuVector(l, false);
	
	float* A = arr;
	float* B = b.arr;
	float* C = c.arr;
	
	for (size_t i = 0; i < l; i++) {
		C[i] = A[i] + B[i];
	}
	
	return c;
}

const CpuVector CpuVector::operator+(float val) const {
	
	size_t l = length;
	CpuVector ans = CpuVector(l, false);
	
	float* A = arr;
	float* B = ans.arr;
	
	for (size_t i = 0; i < l; i++) {
		B[i] = A[i] + val;
	}
	
	return ans;
}

const CpuVector CpuVector::operator-(const CpuVector& b) const {
	check_same_length(b);
	
	size_t l = length;
	CpuVector c = CpuVector(l, false);
	
	float* A = arr;
	float* B = b.arr;
	float* C = c.arr;
	
	for (size_t i = 0; i < l; i++) {
		C[i] = A[i] - B[i];
	}
	
	return c;
}

const CpuVector CpuVector::operator-(float val) const {
	
	size_t l = length;
	CpuVector ans = CpuVector(l, false);
	
	float* A = arr;
	float* B = ans.arr;
	
	for (size_t i = 0; i < l; i++) {
		B[i] = A[i] - val;
	}
	
	return ans;
}

const CpuVector CpuVector::operator-() const {
	
	size_t l = length;
	CpuVector ans = CpuVector(l, false);
	
	float* A = arr;
	float* B = ans.arr;
	
	for (size_t i = 0; i < l; i++) {
		B[i] = -A[i];
	}
	
	return ans;
}

const CpuVector CpuVector::operator*(const CpuVector& b) const {
	check_same_length(b);
	
	size_t l = length;
	CpuVector c = CpuVector(l, false);
	
	float* A = arr;
	float* B = b.arr;
	float* C = c.arr;
	
	for (size_t i = 0; i < l; i++) {
		C[i] = A[i] * B[i];
	}
	
	return c;
}

const CpuVector CpuVector::operator*(float scalar) const {
	
	size_t l = length;
	CpuVector ans = CpuVector(l, false);
	
	float* A = arr;
	float* B = ans.arr;
	
	for (size_t i = 0; i < l; i++) {
		B[i] = A[i] * scalar;
	}
	
	return ans;
}

const CpuVector CpuVector::operator^(float exp) const {
	
	size_t l = length;
	CpuVector ans = CpuVector(l, false);
	
	float* A = arr;
	float* B = ans.arr;
	
	for (size_t i = 0; i < l; i++) {
		B[i] = pow(A[i], exp);
	}
	
	return ans;
}

const CpuVector CpuVector::operator/(const CpuVector& b) const {
	check_same_length(b);
	
	size_t l = length;
	CpuVector c = CpuVector(l, false);
	
	float* A = arr;
	float* B = b.arr;
	float* C = c.arr;
	
	for (size_t i = 0; i < l; i++) {
		C[i] = A[i] / B[i];
	}
	
	return c;
}

const CpuVector CpuVector::operator/(float scalar) const {
	
	size_t l = length;
	CpuVector ans = CpuVector(l, false);
	
	float* A = arr;
	float* B = ans.arr;
	
	for (size_t i = 0; i < l; i++) {
		B[i] = A[i] / scalar;
	}
	
	return ans;
}

float CpuVector::sum() const {
	
	size_t l = length;
	float* A = arr;
	
	float ans = A[0];
	for (size_t i = 1; i < l; i++) {
		ans += A[i];
	}
	
	return ans;
}

float CpuVector::max() const {
	
	size_t l = length;
	float* A = arr;
	
	float ans = A[0];
	for (size_t i = 1; i < l; i++) {
		ans = std::max(ans, A[i]);
	}
	
	return ans;
}

float CpuVector::min() const {
	
	size_t l = length;
	float* A = arr;
	
	float ans = A[0];
	for (size_t i = 1; i < l; i++) {
		ans = std::min(ans, A[i]);
	}
	
	return ans;
}

float CpuVector::avg() const {
	return sum() / length;
}

float CpuVector::var() const {
	
	//from:
	//https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
	size_t n = 0;
	float _mean = 0.0;
	float m2 = 0.0;
	
	float* A = arr;
	size_t l = length;
	for (size_t i = 0; i < l; i++) {
		n++;
		float x = A[i];
		float delta = x - _mean;
		_mean += delta / n;
		m2 += delta * (x - _mean);
		
	}
	
	float ans = (m2 / (n - 1));
	return ans;
}

float CpuVector::stdev() const {
	return sqrt(var());
}

float CpuVector::dot(const CpuVector b) const {
	
	size_t l = length;
	
	float* A = arr;
	float* B = b.arr;
	
	float ans = 0.0;
	for (size_t i = 0; i < l; i++) {
		ans += A[i] * B[i];
	}
	
	return ans;
}

float* CpuVector::ptr() const {
	return arr;
}

void CpuVector::print() {
	size_t l = std::min(VECTOR_PRINT_MAX, length);
	
	if (l > VECTOR_PRINT_MAX) {
		printf("Vector  %d   (truncated)\n", (int) l);
	} else {
		printf("Vector  %d\n", (int) l);
	}
	
	for (size_t i = 0; i < l; i++) {
		printf("%10.4f", arr[i]);
		printf("\n");
	}
	
	printf("\n");
	fflush(stdout);
}

CpuVector::~CpuVector() {
	if (arr) {
		free(arr);
	}
}

} // namespace math
} // namespace cs


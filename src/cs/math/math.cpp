/*
 * math.cpp
 *
 *  Created on: Jan 28, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/core/Exception.h>
#include <cs/math/CpuMatrix.h>
#include <cs/math/CpuVector.h>
#include <cs/math/GpuMatrix.h>
#include <cs/math/GpuVector.h>
#include <cs/math/math.h>
#include <pthread.h>
#include <stddef.h>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <cs/core/lang.h>

namespace cs {
using namespace core;
namespace math {

size_t VECTOR_PRINT_MAX = 100;
size_t MATRIX_PRINT_MAX = 100;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;



bool is_cpu(const Matrix& m){
	return is_cpu(&m);
}

bool is_cpu(const Matrix* m) {
	const CpuMatrix* ans = dynamic_cast<const CpuMatrix*>(m);
	if (ans) {
		return true;
	}
	return false;
}

CpuMatrix& cpu_cast(Matrix& m) {
	const Matrix& c = const_cast<const Matrix&>(m);
	CpuMatrix& ans = const_cast<CpuMatrix&>(cpu_cast(c));
	return ans;
}

CpuMatrix& cpu_cast(Matrix* m) {
	const Matrix* c = const_cast<const Matrix*>(m);
	CpuMatrix& ans = const_cast<CpuMatrix&>(cpu_cast(c));
	return ans;
}

CpuVector& cpu_cast(Vector& v) {
	const Vector& c = const_cast<const Vector&>(v);
	CpuVector& ans = const_cast<CpuVector&>(cpu_cast(c));
	return ans;
}

CpuVector& cpu_cast(Vector* v) {
	const Vector* c = const_cast<const Vector*>(v);
	CpuVector& ans = const_cast<CpuVector&>(cpu_cast(c));
	return ans;
}

const CpuMatrix& cpu_cast(const Matrix& m) {
	return cpu_cast(&m);
}

const CpuMatrix& cpu_cast(const Matrix* m) {
	check_null(m);
	const CpuMatrix* ans = dynamic_cast<const CpuMatrix*>(m);
	if (ans) {
		return *ans;
	}
	throw Exception("Class cast exception. The pointer could not be casted into a CpuMatrix.");
}

const CpuVector& cpu_cast(const Vector& v) {
	return cpu_cast(&v);
}

const CpuVector& cpu_cast(const Vector* v) {
	check_null(v);
	const CpuVector* ans = dynamic_cast<const CpuVector*>(v);
	if (ans) {
		return *ans;
	}
	throw Exception("Class cast exception. The pointer could not be casted into a CpuVector.");
}

bool is_gpu(Matrix& m) {
	return is_gpu(&m);
}

bool is_gpu(Matrix* m) {
	GpuMatrix* ans = dynamic_cast<GpuMatrix*>(m);
	if (ans) {
		return true;
	}
	return false;
}

GpuMatrix& gpu_cast(Matrix& m) {
	const Matrix& c = const_cast<const Matrix&>(m);
	GpuMatrix& ans = const_cast<GpuMatrix&>(gpu_cast(c));
	return ans;
}

GpuMatrix& gpu_cast(Matrix* m) {
	const Matrix* c = const_cast<const Matrix*>(m);
	GpuMatrix& ans = const_cast<GpuMatrix&>(gpu_cast(c));
	return ans;
}

GpuVector& gpu_cast(Vector& v) {
	const Vector& c = const_cast<const Vector&>(v);
	GpuVector& ans = const_cast<GpuVector&>(gpu_cast(c));
	return ans;
}

GpuVector& gpu_cast(Vector* v) {
	const Vector* c = const_cast<const Vector*>(v);
	GpuVector& ans = const_cast<GpuVector&>(gpu_cast(c));
	return ans;
}

const GpuMatrix& gpu_cast(const Matrix& m) {
	return gpu_cast(&m);
}

const GpuMatrix& gpu_cast(const Matrix* m) {
	check_null(m);
	const GpuMatrix* ans = dynamic_cast<const GpuMatrix*>(m);
	if (ans) {
		return *ans;
	}
	throw Exception("Class cast exception. The pointer could not be casted into a GpuMatrix.");
}

const GpuVector& gpu_cast(const Vector& m) {
	return gpu_cast(&m);
}

const GpuVector& gpu_cast(const Vector* m) {
	check_null(m);
	const GpuVector* ans = dynamic_cast<const GpuVector*>(m);
	if (ans) {
		return *ans;
	}
	throw Exception("Class cast exception. The pointer could not be casted into a GpuVector.");
}

float sum(const Matrix& m) {
	return m.sum();
}

const CpuMatrix randn(size_t m, size_t n) {
	CpuMatrix ans = CpuMatrix(m, n, false);
	ans.randn();
	return ans;
}

const CpuVector randn(size_t length) {
	CpuVector ans = CpuVector(length, false);
	
	float* arr = ans.ptr();
	randn(arr, ans.length);
	
	return ans;
}

void randn(float* arr, size_t length) {
	
	for (size_t i = 0; i < length; i++) {
		arr[i] = (float) grandn(0, 1.0);
	}
}

double grandn(double mu, double sigma) {
	
	double ans;
	
	//from https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
	//modified for thread safety
	const double epsilon = std::numeric_limits<double>::min();
	const double two_pi = 2.0 * 3.14159265358979323846;
	
	pthread_mutex_lock(&lock);
	static double z0, z1;
	static bool generate;
	generate = !generate;
	
	if (generate) {
		double u1, u2;
		do {
			u1 = rand() * (1.0 / RAND_MAX);
			u2 = rand() * (1.0 / RAND_MAX);
		} while (u1 <= epsilon);
		
		z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
		z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
		
		ans = z0 * sigma + mu;
	} else {
		ans = z1 * sigma + mu;
	}
	
	pthread_mutex_unlock(&lock);
	return ans;
}

const CpuVector operator*(float scalar, const CpuVector& a) {
	return a * scalar;
}

const CpuMatrix operator*(float scalar, const CpuMatrix& a) {
	return a * scalar;
}

const GpuVector operator*(float scalar, const GpuVector& a) {
	return a * scalar;
}

const GpuMatrix operator*(float scalar, const GpuMatrix& a) {
	return a * scalar;
}

} // namespace math
} // namespace cs

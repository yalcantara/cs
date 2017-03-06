/*
 * errors.cpp
 *
 *  Created on: Feb 25, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/nn/errors.h>
#include <cs/math/math.h>
#include <stdlib.h>

namespace cs {
using namespace math;
namespace nn {

float min_square_error(const Matrix& h, const Matrix& y) {
	if (is_cpu(h)) {
		return min_square_error(cpu_cast(h), cpu_cast(y));
	}
	
	return min_square_error(gpu_cast(h), gpu_cast(y));
}

float min_square_error(const CpuMatrix& h, const CpuMatrix& y) {
	
	size_t m = y.m;
	
	float ans = 1.0 / (2 * m) * sum((h - y) ^ 2);
	
	return ans;
}

float min_square_error(const GpuMatrix& h, const GpuMatrix& y) {
	
	size_t m = y.m;
	
	float ans = 1.0 / (2 * m) * sum((h - y) ^ 2);
	
	return ans;
}

} // namespace nn
} // namespace cs

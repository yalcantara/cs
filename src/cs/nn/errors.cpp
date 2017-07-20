/*
 * errors.cpp
 *
 *  Created on: Feb 25, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/nn/errors.h>
#include <cs/math/math.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;

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
	
	float ans = 0.0f;
	
	size_t m = y.m;
	size_t n = y.n;
	
	for(size_t i =0; i < m; i++){
		for(size_t j =0; j < n; j++){
			ans += pow(h.get(i, j) - y.get(i, j), 2);
		}
	}
	
	ans = ans / (2 * m);
	
	return ans;
}

float min_square_error(const GpuMatrix& h, const GpuMatrix& y) {
	
	size_t m = y.m;
	
	float ans = 1.0 / (2 * m) * sum((h - y) ^ 2);
	
	return ans;
}

} // namespace nn
} // namespace cs

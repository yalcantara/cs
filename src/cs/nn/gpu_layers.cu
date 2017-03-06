/*
 * gpu_layers.cu
 *
 *  Created on: Feb 25, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/math/GpuMatrix.h>
#include <cs/nn/gpu_layers.cuh>
#include <cs/gpu/cuda_utils.cuh>
#include <cs/gpu/gpu.h>

namespace cs {
using namespace math;
using namespace gpu;
namespace nn {


void affine_dx(const GpuMatrix& x, const GpuMatrix& w, const GpuMatrix& dg, GpuMatrix& dx, GpuMatrix& dw,
		GpuVector& db) {
	
	size_t m = x.m;
	size_t n = x.n;
	size_t p = w.n;
	
	float* X = x.ptr();
	float* W = w.ptr();
	float* DG = dg.ptr();
	float* DX = dx.ptr();
	float* DW = dw.ptr();
	float* DB = db.ptr();
	
	gpu_dot(X, true, DG, DW, n, m, p);
	gpu_dot(DG, W, true, DX, m, p, n);
	gpu_sum_rows(DG, DB, m, p);
}

void update_params(const GpuMatrix& w, const GpuMatrix& dw, float scalar) {
	
	size_t length = w.length;
	float* W = w.ptr();
	float* DW = dw.ptr();
	
	gpu_add_inplace(W, DW, scalar, length);
}

void update_params(const GpuVector& b, const GpuVector& db, float scalar) {
	
	size_t length = b.length;
	float* B = b.ptr();
	float* DB = db.ptr();
	
	gpu_add_inplace(B, DB, scalar, length);
}

} // namespace nn
} // namespace cs

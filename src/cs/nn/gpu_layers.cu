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
	size_t o = w.m;
	size_t p = w.n;
	
	float* X = x.ptr();
	float* W = w.ptr();
	float* DG = dg.ptr();
	float* DX = dx.ptr();
	float* DW = dw.ptr();
	float* DB = db.ptr();
	
	gpu_dot(X, true, DG, DW, m, n, p);
	gpu_dot(DG, W, true, DX, m, p, o, p);

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


void sigmoid_fx(const GpuMatrix& x, const GpuMatrix& fx){
	
	size_t m = x.m;
	size_t n = x.n;
	
	float* X = x.ptr();
	float* FX = fx.ptr();
	
	gpu_sigmoid_fx(X, FX, m, n);
}

void sigmoid_dx(const GpuMatrix& x, const GpuMatrix& dx){
	
	size_t m = x.m;
	size_t n = x.n;
	
	float* X = x.ptr();
	float* DX = dx.ptr();
	
	gpu_sigmoid_dx(X, DX, m, n);
}


} // namespace nn
} // namespace cs

/*
 * gpu.cpp
 *
 *  Created on: Jan 30, 2017
 *      Author: Yaison Alcantara
 */

#include "cs/gpu/gpu.h"
#include "cs/core/Exception.h"
#include "cs/gpu/cuda_utils.cuh"

namespace cs {

using namespace core;

namespace gpu {

cublasHandle_t cublas_handle = nullptr;

void init() {
	if (cublas_handle == nullptr) {
		check_cublas(cublasCreate(&cublas_handle));
	}
}

const char* _cuda_get_error_enum(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";
		
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
		
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
		
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
		
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
		
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
		
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
		
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	
	return "<unknown>";
}

void check_cublas(cublasStatus_t status) {
	if (status == CUBLAS_STATUS_SUCCESS) {
		return;
	}
	
	fprintf(stderr, "CUBLAS error %d\nMessage: %s.\n", status, _cuda_get_error_enum(status));
	fflush(stderr);
	throw Exception("Cuda error");
}

void check_cuda(cudaError_t error) {
	if (error == cudaSuccess) {
		return;
	}
	
	const char* name = cudaGetErrorName(error);
	const char* msg = cudaGetErrorString(error);
	fprintf(stderr, "CUDA error: %d %s\nMessage: %s.\n", error, name, msg);
	fflush(stderr);
	throw Exception("Cuda error.");
}

void copy_cpu_to_gpu(float* src, float* dest, size_t length) {
	check_cuda(cudaMemcpy(dest, src, sizeof(float) * length, cudaMemcpyHostToDevice));
}

void copy_gpu_to_cpu(float* src, float* dst, size_t length) {
	check_cuda(cudaMemcpy(dst, src, sizeof(float) * length, cudaMemcpyDeviceToHost));
}

void copy_gpu_to_gpu(float* src, float* dst, size_t length) {
	check_cuda(cudaMemcpy(dst, src, sizeof(float) * length, cudaMemcpyDeviceToDevice));
}

void gpu_free(float* devPtr) {
	check_cuda(cudaFree(devPtr));
}

void gpu_set(float* a, float val, size_t l) {
	check_cuda(cudaMemset(a, val, sizeof(float) * l));
}

float* gpu_malloc(size_t length, bool clear) {
	
	if (length < 1) {
		throw Exception("Invalid length " + to_string(length) + ".");
	}
	
	float* devPtr;
	check_cuda(cudaMalloc(&devPtr, sizeof(float) * length));
	if (clear) {
		gpu_set(devPtr, 0, length);
	}
	
	return devPtr;
}

void gpu_add(float* a, float* b, float* c, size_t l) {
	init();
	
	copy_gpu_to_gpu(a, c, l);
	const float alpha = 1;
	check_cublas(cublasSaxpy(cublas_handle, l, &alpha, b, 1, c, 1));
}

void gpu_add_inplace(float* a, float* b, size_t l) {
	init();
	
	const float alpha = 1;
	check_cublas(cublasSaxpy(cublas_handle, l, &alpha, b, 1, a, 1));
}

void gpu_add_inplace(float* a, float* b, const float alpha, size_t l) {
	init();
	
	check_cublas(cublasSaxpy(cublas_handle, l, &alpha, b, 1, a, 1));
}

void gpu_sub(float* a, float* b, float* c, size_t l) {
	init();
	
	copy_gpu_to_gpu(a, c, l);
	const float alpha = -1;
	check_cublas(cublasSaxpy(cublas_handle, l, &alpha, b, 1, c, 1));
}

void gpu_sub_inplace(float* a, float* b, size_t l) {
	init();
	
	const float alpha = -1;
	check_cublas(cublasSaxpy(cublas_handle, l, &alpha, b, 1, a, 1));
}

void gpu_mult(float* src, float scalar, float* dest, size_t l) {
	init();
	
	copy_gpu_to_gpu(src, dest, l);
	check_cublas(cublasSscal(cublas_handle, l, &scalar, dest, 1));
}

void gpu_mult_inplace(float* src, float scalar, size_t l) {
	init();
	
	check_cublas(cublasSscal(cublas_handle, l, &scalar, src, 1));
}

void gpu_mult(float* a, float* b, float* c, size_t l) {
	init();
	cuda_vector_mult(a, b, c, l);
}

void gpu_mult_inplace(float* a, float* b, size_t l) {
	init();
	cuda_vector_mult(a, b, a, l);
}

void gpu_div(float* a, float* b, float* c, size_t l) {
	init();
	cuda_vector_div(a, b, c, l);
}

void gpu_div_inplace(float* a, float* b, size_t l) {
	init();
	cuda_vector_div(a, b, a, l);
}

void gpu_pow(float* src, float exp, float* dest, size_t l) {
	init();
	cuda_vector_pow(src, exp, dest, l);
}

void gpu_pow_inplace(float* src, float exp, size_t l) {
	init();
	cuda_vector_pow(src, exp, src, l);
}

float gpu_sum(float* a, size_t l) {
	init();
	float val;
	check_cublas(cublasSasum(cublas_handle, l, a, 1, &val));
	
	return val;
}

void gpu_dot(float* a, float* b, float* c, size_t m, size_t n, size_t p) {
	gpu_dot(a, false, b, c, m, n, p);
}

void gpu_dot(float* a, bool transA, float* b, float* c, size_t m, size_t n, size_t p) {
	init();
	const float alpha = 1.0;
	const float beta = 0.0;
	
	//Since cublas assumes column major, and our structure are row major, we need to change the order.
	// The logic is that (AB)^T = B^T x A^T.
	if (transA) {
		check_cublas(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, p, m, n, &alpha, b, p, a, m, &beta, c, p));
	} else {
		check_cublas(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, p, m, n, &alpha, b, p, a, n, &beta, c, p));
	}
}


void gpu_dot(float* a, float* b, bool transB, float* c, size_t m, size_t n, size_t p) {
	init();
	const float alpha = 1.0;
	const float beta = 0.0;
	
	//Since cublas assumes column major, and our structure are row major, we need to change the order.
	// The logic is that (AB)^T = B^T x A^T.
	if (transB) {
		check_cublas(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, p, m, n, &alpha, b, n, a, p, &beta, c, n));
	} else {
		check_cublas(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, p, m, n, &alpha, b, p, a, n, &beta, c, p));
	}
}



void gpu_dot(float* a, float* b, float* c, size_t m, size_t n) {
	init();
	const float alpha = 1.0;
	const float beta = 0.0;
	
	//Since cublas assumes column major, and our structure are row major, we need to transpose the matrix.
	check_cublas(cublasSgemv(cublas_handle, CUBLAS_OP_T, n, m, &alpha, a, n, b, 1, &beta, c, 1));
}

void gpu_broadcast_sum_rows(float* a, float* b, float* c, size_t m, size_t n) {
	cuda_broadcast_sum_rows(a, b, c, m, n);
}

void gpu_sum_rows(float* a, float* dest, size_t m, size_t n){
	cuda_sum_rows(a, dest, m, n);
}

} // namespace gpu
} // namespace cs

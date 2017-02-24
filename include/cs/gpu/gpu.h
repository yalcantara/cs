/*
 * gpu.h
 *
 *  Created on: Jan 28, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_GPU_GPU_H_
#define CS_GPU_GPU_H_

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cs/core/lang.h"


namespace cs{

namespace gpu{

void check_cuda(cudaError_t error);
void check_cublas(cublasStatus_t status);
const char* _cuda_get_error_enum(cublasStatus_t error);
void copy_cpu_to_gpu(float* src, float* dest, size_t length);
void copy_gpu_to_cpu(float* src, float* dst, size_t length);
void copy_gpu_to_gpu(float* src, float* dst, size_t length);

void gpu_free(float* devPtr);
float* gpu_malloc(size_t length, bool clear);

void gpu_add(float* a, float* b, float* c, size_t l);
void gpu_add_inplace(float* a, float* b, size_t l);
void gpu_sub(float* a, float* b, float* c, size_t l);
void gpu_sub_inplace(float* a, float* b, size_t l);
void gpu_sub(float* a, float* b, float* c, size_t l);
void gpu_sub_inplace(float* a, float* b, size_t l);

void gpu_mult(float* src, float scalar, float* dest, size_t l);
void gpu_mult_inplace(float* src, float scalar, size_t l);
void gpu_mult(float* a, float* b, float* c, size_t l);
void gpu_mult_inplace(float* a, float* b, size_t l);
void gpu_div(float* a, float* b, float* c, size_t l);
void gpu_div_inplace(float* a, float* b, size_t l);
void gpu_pow(float* src, float exp, float* dest, size_t l);
void gpu_pow_inplace(float* src, float exp, size_t l);

float gpu_sum(float* a, size_t l);

void gpu_dot(float* a, float* b, float* c, size_t m, size_t n, size_t p);
void gpu_dot(float* a, float* b, float* c, size_t m, size_t n);
void gpu_broadcast_sum_rows(float* a, float* b, float* c, size_t m, size_t n);


//activation
void gpu_sigmoid_fx(float* x, float* fx, size_t l);
void gpu_sigmoid_dx(float* fx, float* dx, size_t l);
void gpu_relu_fx(float* x, float* fx, size_t l);
void gpu_relu_dx(float* fx, float* dx, size_t l);
void gpu_2dconv_fx(float* x, float* w, float* fx, size_t m, size_t n, size_t d);
void gpu_2dconv_dx(float* x, float* w, float* dx, float* dw, size_t m, size_t n, size_t d);

}// namespace gpu
}// namespace cs

#endif // CS_GPU_GPU_H_

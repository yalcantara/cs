/*
 * cuda_utils.cuh
 *
 *  Created on: Feb 4, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_GPU_CUDA_UTILS_CUH_
#define CS_GPU_CUDA_UTILS_CUH_


namespace cs{
namespace gpu{


extern unsigned int BLOCK_SIZE_2D;
extern unsigned int BLOCK_SIZE_1D;

void cuda_matrix_mult(float* a, float* b, float* dest, size_t m, size_t n);
void cuda_vector_mult(float* a, float* b, float* dest, size_t length);
void cuda_vector_div(float* a, float* b, float* dest, size_t length);
void cuda_vector_pow(float* a, float exponent, float* dest, size_t l);

void cuda_broadcast_sum_rows(float* a, float* b, float* dest, size_t m, size_t n);
void cuda_sum_rows(float* a, float* dest, size_t m, size_t n);

void cuda_sigmoid_fx(float* x, float* fx, size_t m, size_t n);
void cuda_sigmoid_dx(float* x, float* fx, size_t m, size_t n);

} // namespace gpu
} // namespace cs
#endif // CS_GPU_CUDA_UTILS_CUH_

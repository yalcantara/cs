/*
 * cuda_utils.cu
 *
 *  Created on: Feb 4, 2017
 *      Author: Yaison Alcantara
 */

namespace cs {
namespace gpu {

unsigned int BLOCK_SIZE_2D = 16;
unsigned int BLOCK_SIZE_1D = 256;

__global__ void cuda_kernel_matrix_mult(float* a, float* b, float* dest, unsigned int m, unsigned int n) {
	
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < n) {
		unsigned int absIdx = i * n + j;
		dest[absIdx] = a[absIdx] * b[absIdx];
	}
}

__global__ void cuda_kernel_matrix_sum_rows(float* a, float* dest, unsigned int m, unsigned int n) {
	
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < n) {
		float sum = 0.0;
		for (unsigned int i = 0; i < m; i++) {
			sum += a[i * n + j];
		}
		dest[j] = sum;
	}
}

__global__ void cuda_kernel_vector_mult(float* a, float* b, float* dest, unsigned int l) {
	
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < l) {
		dest[idx] = a[idx] * b[idx];
	}
}

__global__ void cuda_kernel_vector_div(float* a, float* b, float* dest, unsigned int l) {
	
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < l) {
		dest[idx] = a[idx] / b[idx];
	}
}

__global__ void kernel_vector_pow(float* a, float exp, float* dest, unsigned int l) {
	
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < l) {
		dest[idx] = powf(a[idx], exp);
	}
}

__global__ void kernel_broadcast_sum_rows(float* a, float* b, float* dest, unsigned int m, unsigned int n) {
	
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < n) {
		unsigned int absIdx = i * n + j;
		dest[absIdx] = a[absIdx] + b[j];
	}
}

void cuda_matrix_mult(float* a, float* b, float* dest, size_t m, size_t n) {
	
	dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
	
	unsigned int blocksX = (unsigned int) ceil(n / (double) BLOCK_SIZE_2D);
	unsigned int blocksY = (unsigned int) ceil(m / (double) BLOCK_SIZE_2D);
	
	dim3 grid(blocksX, blocksY);
	
	cuda_kernel_matrix_mult<<<grid, block>>>(a, b, dest, m, n);
}

void cuda_vector_mult(float* a, float* b, float* dest, size_t length) {
	
	dim3 block(BLOCK_SIZE_1D);
	
	unsigned int blocksX = (unsigned int) ceil(length / (double) BLOCK_SIZE_1D);
	
	dim3 grid(blocksX);
	
	cuda_kernel_vector_mult<<<grid, block>>>(a, b, dest, length);
}

void cuda_vector_div(float* a, float* b, float* dest, size_t length) {
	
	dim3 block(BLOCK_SIZE_1D);
	
	unsigned int blocksX = (unsigned int) ceil(length / (double) BLOCK_SIZE_1D);
	
	dim3 grid(blocksX);
	
	cuda_kernel_vector_div<<<grid, block>>>(a, b, dest, length);
}

void cuda_vector_pow(float* a, float exponent, float* dest, size_t l) {
	
	dim3 block(BLOCK_SIZE_1D);
	
	unsigned int blocksX = (unsigned int) ceil(l / (double) BLOCK_SIZE_1D);
	
	dim3 grid(blocksX);
	
	kernel_vector_pow<<<grid, block>>>(a, exponent, dest, l);
}

void cuda_broadcast_sum_rows(float* a, float* b, float* dest, size_t m, size_t n) {
	
	dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
	
	unsigned int blocksX = (unsigned int) ceil(n / (double) BLOCK_SIZE_2D);
	unsigned int blocksY = (unsigned int) ceil(m / (double) BLOCK_SIZE_2D);
	
	dim3 grid(blocksX, blocksY);
	
	kernel_broadcast_sum_rows<<<grid, block>>>(a, b, dest, m, n);
}

void cuda_sum_rows(float* a, float* dest, size_t m, size_t n) {
	
	dim3 block(BLOCK_SIZE_1D);
	//Note: in this case is n as the number of threads to use, cuz each thread is doing a reduce operation on the rows
	unsigned int blocksX = (unsigned int) ceil(n / (double) BLOCK_SIZE_1D);
	
	dim3 grid(blocksX);
	
	cuda_kernel_matrix_sum_rows<<<grid, block>>>(a, dest, m, n);
}

}
 // namespace gpu
}// namespace cs

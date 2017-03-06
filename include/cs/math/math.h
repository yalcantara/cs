/*
 * math.h
 *
 *  Created on: Jan 28, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_MATH_MATH_H_
#define CS_MATH_MATH_H_

#include <cs/math/CpuMatrix.h>
#include <cs/math/CpuVector.h>
#include <cs/math/GpuMatrix.h>
#include <cs/math/GpuVector.h>
#include <stddef.h>

namespace cs {
namespace math {

extern size_t VECTOR_PRINT_MAX;
extern size_t MATRIX_PRINT_MAX;

bool is_cpu(const Matrix& m);
bool is_cpu(const Matrix* m);
CpuMatrix& cpu_cast(Matrix& m);
CpuMatrix& cpu_cast(Matrix* m);
CpuVector& cpu_cast(Vector& m);
CpuVector& cpu_cast(Vector* m);
const CpuMatrix& cpu_cast(const Matrix& m);
const CpuMatrix& cpu_cast(const Matrix* m);
const CpuVector& cpu_cast(const Vector& m);
const CpuVector& cpu_cast(const Vector* m);

bool is_gpu(Matrix& m);
bool is_gpu(Matrix* m);
GpuMatrix& gpu_cast(Matrix& m);
GpuMatrix& gpu_cast(Matrix* m);
GpuVector& gpu_cast(Vector& m);
GpuVector& gpu_cast(Vector* m);
const GpuMatrix& gpu_cast(const Matrix& m);
const GpuMatrix& gpu_cast(const Matrix* m);
const GpuVector& gpu_cast(const Vector& m);
const GpuVector& gpu_cast(const Vector* m);

const CpuMatrix randn(size_t m, size_t n);
const CpuVector randn(size_t length);
void randn(float* arr, size_t length);
double grandn(double mu, double sigma);

float sum(const Matrix& m);

const CpuVector operator*(float scalar, const CpuVector& a);
const CpuMatrix operator*(float scalar, const CpuMatrix& a);

const GpuVector operator*(float scalar, const GpuVector& a);
const GpuMatrix operator*(float scalar, const GpuMatrix& a);

} // namespace math
} // namespace cs

#endif // CS_MATH_MATH_H_

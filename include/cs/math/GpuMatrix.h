/*
 * GpuMatrix.h
 *
 *  Created on: Feb 9, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_MATH_GPUMATRIX_H_
#define CS_MATH_GPUMATRIX_H_

#include <cs/math/CpuMatrix.h>
#include <cs/math/GpuVector.h>
#include <stddef.h>

namespace cs {
namespace math {

class GpuMatrix: public Matrix {
private:
	float* devPtr;

public:
	GpuMatrix(size_t m, size_t n);
	GpuMatrix(size_t m, size_t n, bool clear);
	GpuMatrix(const CpuMatrix& other);
	GpuMatrix(const GpuMatrix& other);
	GpuMatrix(const initializer_list<const initializer_list<float>> &list);

	GpuMatrix& operator=(const GpuMatrix& other);

	void clear();
	void randn();
	
	const GpuMatrix operator+(const GpuMatrix& b) const;
	const GpuMatrix operator-(const GpuMatrix& b) const;
	const GpuMatrix operator-() const;
	const GpuMatrix operator*(const GpuMatrix& b) const;
	const GpuMatrix operator*(const float scalar) const;
	const GpuMatrix operator/(const GpuMatrix& b) const;
	const GpuMatrix operator/(const float scalar) const;
	const GpuMatrix operator^(const float exp) const;

	void addi(const GpuMatrix& b);
	void subi(const GpuMatrix& b);
	void multi(const GpuMatrix& b);
	void multi(const float scalar);
	void divi(const GpuMatrix& b);
	void divi(const float scalar);
	void powi(const float exp);

	const GpuMatrix dot(const GpuMatrix& b) const;
	void dot(const GpuMatrix& b, GpuMatrix& ans)const;
	
	const GpuVector dot(const GpuVector& b) const;
	const GpuMatrix affine(const GpuMatrix& x, const GpuVector& b)const;
	
	void affine(const Matrix& x, const Vector& b, Matrix& ans)const;
	void affine(const GpuMatrix& x, const GpuVector& b, GpuMatrix& ans) const;

	
	float sum() const;
	

	bool equals(const GpuMatrix& other, float e);
	void copy(Matrix& dest)const;
	const CpuMatrix cpu() const;
	float* ptr() const;
	virtual void print() const;

	virtual ~GpuMatrix();
};

} // namespace math
} // namespace cs

#endif // CS_MATH_GPUMATRIX_H_

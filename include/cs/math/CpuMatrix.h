/*
 * CpuMatrix.h
 *
 *  Created on: Feb 5, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_MATH_CPUMATRIX_H_
#define CS_MATH_CPUMATRIX_H_

#include <cs/math/CpuVector.h>
#include <cs/math/Matrix.h>
#include <stddef.h>


namespace cs {
namespace math {

class CpuMatrix:public Matrix {
	
private:
	float* arr;
	
public:

	CpuMatrix(size_t m, size_t n);
	CpuMatrix(size_t m, size_t n, bool clear);
	CpuMatrix(size_t m, size_t n, float* src);
	CpuMatrix(const CpuMatrix& other);
	CpuMatrix(const initializer_list<const initializer_list<float>> &list);

	void randn();
	void clear();
	
	CpuMatrix& operator=(const CpuMatrix& other);
	float at(size_t idx)const;
	float get(size_t i, size_t j)const;
	void set(size_t i, size_t j, float val)const;
	
	void subi(const CpuMatrix& b);
	
	const CpuMatrix operator+(const CpuMatrix& b)const;
	const CpuMatrix operator+(float val)const;
	const CpuMatrix operator-(const CpuMatrix& b)const;
	const CpuMatrix operator-(float val)const;
	const CpuMatrix operator-()const;
	const CpuMatrix operator*(const CpuMatrix& b)const;
	const CpuMatrix operator*(float scalar)const;
	const CpuMatrix operator/(const CpuMatrix& b)const;
	const CpuMatrix operator/(float scalar)const;
	const CpuMatrix operator^(float exp)const;

	void dot(const CpuMatrix& b, CpuMatrix& ans)const;
	const CpuMatrix dot(const CpuMatrix& b)const;
	const CpuVector dot(const CpuVector& b) const;
	const CpuMatrix affine(const CpuMatrix& x, const CpuVector& b)const;
	
	void affine(const Matrix& x, const Vector& b, Matrix& ans)const;
	void affine(const CpuMatrix& x, const CpuVector& b, CpuMatrix& ans)const;
	
	
	
	float sum()const;
	float max()const;
	float min()const;
	float avg()const;

	void copy(Matrix& dest)const;
	void copy(CpuMatrix& dest)const;
	
	const CpuMatrix sltcols(size_t start, size_t end)const;
	
	
	float* ptr()const;

	void print()const;
	
	
	
	
	virtual ~CpuMatrix();
};

} // namespace math
} // namespace cs

#endif // CS_MATH_CPUMATRIX_H_

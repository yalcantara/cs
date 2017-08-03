/*
 * Matrix.h
 *
 *  Created on: Feb 9, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_MATH_MATRIX_H_
#define CS_MATH_MATRIX_H_

#include <stdlib.h>
#include <cs/math/Vector.h>

namespace cs {
namespace math {

class Matrix {
	
protected:
	void check_dimensions() const;
	void check_index(size_t idx) const;
	void check_index(size_t row, size_t col) const;
	void assert_rows(size_t val, size_t expected) const;
	void assert_cols(size_t val, size_t expected) const;
	

public:
	const size_t m;
	const size_t n;
	const size_t length;
	
	Matrix(size_t m, size_t n);
	
	void check_same_dimensions(const Matrix& other) const;
	
	virtual void clear()=0;
	virtual void affine(const Matrix& x, const Vector& b, Matrix& ans)const=0;
	
	virtual void randn()=0;
	virtual float sum()const=0;
	virtual void copy(Matrix& dest)const=0;
	virtual void print()const=0;
	virtual ~Matrix();
};

} // namespace math
} // namespace cs

#endif // MATRIX_H_

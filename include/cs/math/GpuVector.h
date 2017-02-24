/*
 * GpuVector.h
 *
 *  Created on: Jan 28, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_MATH_GPUVECTOR_H_
#define CS_MATH_GPUVECTOR_H_

#include <cs/math/CpuVector.h>
#include <cs/math/Vector.h>
#include <stddef.h>

using namespace std;

namespace cs {
namespace math {
class GpuVector:public Vector {
	
private:
	float* devPtr;
	void check_same_length(const GpuVector& b)const;

public:
	const size_t length;

	GpuVector(size_t length);
	GpuVector(size_t length, bool clear);
	GpuVector(const CpuVector& other);
	GpuVector(const GpuVector& other);
	GpuVector(const initializer_list<float> &list);

	GpuVector& operator=(const GpuVector& other);

	const GpuVector operator+(const GpuVector& b) const;
	const GpuVector operator+(const float val) const;
	const GpuVector operator-(const GpuVector& b) const;
	const GpuVector operator-(const float val) const;
	const GpuVector operator-() const;
	const GpuVector operator*(const GpuVector& b) const;
	const GpuVector operator*(const float scalar) const;
	const GpuVector operator/(const GpuVector& b) const;
	const GpuVector operator/(const float scalar) const;
	const GpuVector operator^(const float exp) const;

	void addi(const GpuVector& b);
	void subi(const GpuVector& b);
	void multi(const GpuVector& b);
	void multi(const float scalar);
	void divi(const GpuVector& b);
	void divi(const float scalar);
	void powi(const float exp);
	
	float sum()const;
	
	const CpuVector cpu() const;
	void randn();
	float* ptr() const;
	void print() const;

	virtual ~GpuVector();
};
} //namespace math
} //namespace cs

#endif /* CS_MATH_GPUVECTOR_H_ */

/*
 * CpuVector.h
 *
 *  Created on: Jan 28, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_MATH_CPUVECTOR_H_
#define CS_MATH_CPUVECTOR_H_

#include <stdlib.h>
#include <initializer_list>
#include <cs/math/Vector.h>

using namespace std;

namespace cs{
namespace math{


class CpuVector:public Vector {

private:
	float* arr;
	
	void check_index(size_t idx)const;
	void check_same_length(const CpuVector& other)const;

public:
	const size_t length;

	CpuVector(size_t length);
	CpuVector(size_t length, bool clear);
	CpuVector(const CpuVector& other);
	CpuVector(const initializer_list<float> &list);

	CpuVector& operator=(const CpuVector& other);
	float operator[](size_t idx)const;
	float& operator[](size_t idx);
	
	const CpuVector operator+(const CpuVector& b)const;
	const CpuVector operator+(float val)const;
	const CpuVector operator-(const CpuVector& b)const;
	const CpuVector operator-(float val)const;
	const CpuVector operator-()const;
	const CpuVector operator*(const CpuVector& b)const;
	const CpuVector operator*(float scalar)const;
	const CpuVector operator/(const CpuVector& b)const;
	const CpuVector operator/(float scalar)const;
	const CpuVector operator^(float exp)const;
		
	float sum()const;
	float max()const;
	float min()const;
	float avg()const;
	float var()const;
	float stdev()const;
	
	float dot(const CpuVector b)const;

	void randn();
	float* ptr()const;

	void print();
	virtual ~CpuVector();
};
} // namespace math
} // namespace cs
#endif // CS_MATH_CPUVECTOR_H_

/*
 * Vector.h
 *
 *  Created on: Feb 20, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_MATH_VECTOR_H_
#define CS_MATH_VECTOR_H_

#include <stdlib.h>

namespace cs {
namespace math {

class Vector {
protected:
	
	void check_index(size_t idx) const;
	void check_same_length(const Vector& other) const;

public:
	
	const size_t length;

	Vector(size_t length);

	virtual void randn()=0;
	virtual void clear()=0;
	virtual void copy(Vector& dest)const=0;
	virtual void print()const=0;
	virtual ~Vector();
};

} // namespace math
} // namespace cs

#endif // CS_MATH_VECTOR_H_

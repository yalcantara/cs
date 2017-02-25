/*
 * Vector.h
 *
 *  Created on: Feb 20, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_MATH_VECTOR_H_
#define CS_MATH_VECTOR_H_

namespace cs {
namespace math {

class Vector {
public:
	Vector();
	
	virtual void randn()=0;
	virtual void clear()=0;
	virtual ~Vector();
};

} // namespace math
} // namespace cs

#endif // CS_MATH_VECTOR_H_

/*
 * RowView.h
 *
 *  Created on: Feb 5, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_MATH_ROWVIEW_H_
#define CS_MATH_ROWVIEW_H_

#include <stdlib.h>


namespace cs {
namespace math {

template<typename T>
class RowView {
public:
	RowView();
	
	const T& operator[](size_t idx)const;
	
	virtual ~RowView();
};

} // namespace math 
} // namespace cs 

#endif // CS_MATH_ROWVIEW_H_

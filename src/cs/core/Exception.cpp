/*
 * Exception.cpp
 *
 *  Created on: May 18, 2016
 *      Author: Yaison Alcantara
 */

#include "cs/core/Exception.h"
#include <string>

namespace cs {
namespace core{
Exception::Exception(string msg):msg(msg) {
	
}


const char* Exception::what() const throw(){
	return msg.c_str();
}


Exception::~Exception() {
	// TODO Auto-generated destructor stub
}

} // namespace core
} // namespace cs

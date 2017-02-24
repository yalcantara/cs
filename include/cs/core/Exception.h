/*
 * Exception.h
 *
 *  Created on: May 18, 2016
 *      Author: Yaison Alcantara
 */

#ifndef CS_CORE_EXCEPTION_H_
#define CS_CORE_EXCEPTION_H_

#include <exception>
#include <string>

using namespace std;

namespace cs {
namespace core{

class Exception: public exception {
private:
	const string msg;
public:
	Exception(string msg);
	virtual const char* what() const throw();
	virtual ~Exception();
};

} // namespace core
} // namespace cs
#endif // CS_CORE_EXCEPTION_H_

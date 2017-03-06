/*
 * Vector.cpp
 *
 *  Created on: Feb 20, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/math/Vector.h>
#include <cs/core/Exception.h>

namespace cs {
using namespace core;
namespace math {


void Vector::check_index(size_t idx) const {
	if (idx >= length) {
		throw Exception(
				"Index out of bounds. Expected < " + to_string(length) + ", but got: " + to_string(idx) + " instead.");
	}
}

void Vector::check_same_length(const Vector& other) const {
	if (other.length != length) {
		throw Exception(
				"The length must be the same. Expected " + to_string(length) + ", but got: " + to_string(other.length)
						+ " instead.");
	}
}


Vector::Vector(size_t length):length(length) {
	
}

Vector::~Vector() {
	
}

} // namespace math
} // namespace cs

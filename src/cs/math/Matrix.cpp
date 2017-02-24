/*
 * Matrix.cpp
 *
 *  Created on: Feb 9, 2017
 *      Author: Yaison Alcantara
 */

#include "cs/math/Matrix.h"
#include "cs/core/Exception.h"

namespace cs {
using namespace core;
namespace math {

Matrix::Matrix(size_t m, size_t n) :
		m(m), n(n), length(m * n) {
	check_dimensions();
}

void Matrix::check_dimensions() const {
	if (m < 1 || n < 1) {
		throw Exception("Invalid dimensions " + to_string(m) + "x" + to_string(n) + ".");
	}
	
	if (length < 1) {
		throw Exception("Invalid length " + to_string(length) + ".");
	}
}

void Matrix::check_index(size_t idx) const {
	if (idx < 0 || idx >= length) {
		throw Exception(
				"The absolute index is out of bounds. Allowed ranges [0, " + to_string(length - 1) + "], but got: "
						+ to_string(idx) + " instead.");
	}
}

void Matrix::check_index(size_t row, size_t col) const {
	if (row < 0 || row >= m) {
		throw Exception(
				"The row index is out of bounds. Allowed ranges [0, " + to_string(m - 1) + "], but got: "
						+ to_string(row) + " instead.");
	}
	
	if (col < 0 || col >= n) {
		throw Exception(
				"The column index is out of bounds. Allowed ranges [0, " + to_string(n - 1) + "], but got: "
						+ to_string(col) + " instead.");
	}
}

void Matrix::assert_rows(size_t val, size_t expected) const {
	if (val != expected) {
		throw Exception(
				"The rows must be the same. Expected " + to_string(expected) + ", but got: " + to_string(val)
						+ " instead.");
	}
}

void Matrix::assert_cols(size_t val, size_t expected) const {
	if (val != expected) {
		throw Exception(
				"The columns must be the same. Expected " + to_string(expected) + ", but got: " + to_string(val)
						+ " instead.");
	}
}

void Matrix::check_same_dimensions(const Matrix& other) const {
	assert_rows(other.m, m);
	assert_cols(other.n, n);
}

Matrix::~Matrix() {
	
}

} // namespace math
} // namespace cs

/*
 * errors.hpp
 *
 *  Created on: Feb 25, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_NN_ERRORS_HPP_
#define CS_NN_ERRORS_HPP_

#include <cs/math/Matrix.h>
#include <cs/math/CpuMatrix.h>
#include <cs/math/GpuMatrix.h>

namespace cs {
using namespace math;
namespace nn {



float min_square_error(const Matrix& h, const Matrix& y);
float min_square_error(const CpuMatrix& h, const CpuMatrix& y);
float min_square_error(const GpuMatrix& h, const GpuMatrix& y);
	

} // namespace nn
} // namespace cs
#endif // CS_NN_ERRORS_HPP_

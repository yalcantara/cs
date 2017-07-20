/*
 * MinSquare.h
 *
 *  Created on: Jun 25, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_MATH_MINSQUARE_H_
#define CS_MATH_MINSQUARE_H_

#include <cs/math/CpuMatrix.h>
#include <cs/math/GpuMatrix.h>
#include <cs/nn/Layer.h>
#include <stddef.h>

namespace cs {
namespace nn {

class MinSquare: public nn::Layer {
	
private:
	
	Matrix* x = nullptr; //not owned
	
	void gpu_foward(const GpuMatrix& x, const GpuMatrix& fx);
	void gpu_backward(const GpuMatrix& dg);

	void cpu_foward(const CpuMatrix& x, const CpuMatrix& fx);
	void cpu_backward(const CpuMatrix& dg);

public:
	MinSquare();

	void init();
	void set_dim(size_t inout);

	Matrix& foward(const Matrix& x);
	Matrix& backward(const Matrix& dg);

	void update(float alpha);

	void print() const;

	virtual ~MinSquare();
};

} // namespace math
} // namespace cs 

#endif // CS_MATH_MINSQUARE_H_

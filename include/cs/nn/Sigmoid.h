/*
 * Sigmoid.h
 *
 *  Created on: Mar 6, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_NN_SIGMOID_H_
#define CS_NN_SIGMOID_H_

#include <cs/math/GpuMatrix.h>
#include <cs/math/CpuMatrix.h>
#include <cs/nn/Layer.h>

namespace cs {
using namespace math;
namespace nn {

class Sigmoid: public Layer {
	
private:
	Matrix* x = nullptr; //not owned
	
	
	void gpu_foward(const GpuMatrix& x, const GpuMatrix& fx);
	void gpu_backward(const GpuMatrix& dg);
	
	void cpu_foward(const CpuMatrix& x, const CpuMatrix& fx);
	void cpu_backward(const CpuMatrix& dg);
	
	float cpu_sigmoid_fx(float z);
	float cpu_sigmoid_dx(float z);
	
	
public:
	Sigmoid();
	
	void init();
	void set_dim(size_t inout);

	Matrix& foward(const Matrix& x);
	Matrix& backward(const Matrix& dg);

	void update(float alpha);
	
	

	void print() const;

	virtual ~Sigmoid();
};

} // namespace math
} // namespace cs

#endif // CS_NN_SIGMOID_H_

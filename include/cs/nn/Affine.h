/*
 * Affine.h
 *
 *  Created on: Feb 19, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_NN_AFFINE_H_
#define CS_NN_AFFINE_H_

#include <cs/math/CpuMatrix.h>
#include <cs/math/GpuMatrix.h>
#include <cs/math/Vector.h>
#include <cs/nn/Layer.h>

namespace cs {
using namespace math;
namespace nn {

class Affine: public Layer {
	
private:
	Matrix* x = nullptr;
	Matrix* w = nullptr;
	Vector* b = nullptr;

	Matrix* fx = nullptr;
	Matrix* dx = nullptr;
	Matrix* dw = nullptr;
	Vector* db = nullptr;

	
	Matrix& gpu_backward(const GpuMatrix& dg);
	Matrix& cpu_backward(const CpuMatrix& dg);
	
	void gpu_update(float alpha);
	void cpu_update(float alpha);

public:
	
	Affine();

	void release();
	void init();
	
	void set_weights(const Matrix& weights);
	void set_bias(const Vector& bias);
	
	
	
	Matrix& foward(const Matrix& x);
	Matrix& backward(const Matrix& dg);

	void update(float alpha);

	void print()const;
	
	virtual ~Affine();
};

} // namespace nn
} // namespace cs 

#endif // CS_NN_AFFINE_H_

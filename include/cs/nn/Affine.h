/*
 * Affine.h
 *
 *  Created on: Feb 19, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_NN_AFFINE_H_
#define CS_NN_AFFINE_H_

#include <cs/nn/Layer.h>
#include <cs/math/Matrix.h>
#include <cs/math/Vector.h>

namespace cs {
using namespace math;
namespace nn {

class Affine:public Layer {
	
private:

	Matrix* w = nullptr;
	Matrix* x = nullptr;
	Vector* b = nullptr;
	
	
	Matrix* fx = nullptr;
	Matrix* dx = nullptr;
	Matrix* dw = nullptr;
	Vector* db = nullptr;
	
	void init_fx(size_t m);
	void init_dx(size_t m, size_t n);
	
	Matrix& cpu_backward(Matrix& dg);
	
public:
	Affine();
	
	void release();
	void init();
	
	Matrix& foward(Matrix& x);
	Matrix& backward(Matrix& dg);

	void update(float alpha);
	
	virtual ~Affine();
};

} // namespace nn
} // namespace cs 

#endif // CS_NN_AFFINE_H_

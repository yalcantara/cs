/*
 * Layer.h
 *
 *  Created on: Feb 19, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_NN_LAYER_H_
#define CS_NN_LAYER_H_

#include <cs/math/Matrix.h>


namespace cs {
using namespace math;
namespace nn {

class Layer {
protected:
	bool gpu = false;
	size_t in = 0;
	size_t out = 0;

	Matrix* fx = nullptr;
	Matrix* dx = nullptr;

	void init_fx(size_t m);
	void init_dx(size_t m, size_t n);

public:
	Layer();

	void use_gpu(bool val);

	void set_dim(size_t input, size_t output);
	virtual void init()=0;

	void set_in_dim(size_t in);
	void set_out_dim(size_t out);
	
	size_t in_dim() const;
	size_t out_dim() const;

	Matrix& get_dx() const;
	Matrix& get_fx()const;
	
	bool has_fx()const;

	virtual Matrix& foward(const Matrix& x)=0;
	virtual Matrix& backward(const Matrix& dg)=0;
	virtual void update(float alpha)=0;

	virtual void print() const=0;
	virtual ~Layer();
};

} // namespace math
} // namespace cs

#endif // CS_NN_LAYER_H_

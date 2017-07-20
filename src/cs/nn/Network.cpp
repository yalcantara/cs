/*
 * Network.cpp
 *
 *  Created on: Jun 24, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/nn/Network.h>
#include <cs/core/lang.h>
#include <cs/core/Exception.h>
#include <cs/math/math.h>
#include <cs/nn/errors.h>

namespace cs {
using namespace core;
using namespace math;
namespace nn {

Network::Network() {
	// TODO Auto-generated constructor stub
	
}

void Network::operator<<(Affine layer) {
	Affine* l = new Affine();
	
	layers.push_back(l);
}

void Network::operator<<(Sigmoid layer) {
	Sigmoid* l = new Sigmoid();
	
	layers.push_back(l);
}

void Network::operator<<(MinSquare layer) {
	
}

void Network::init(Matrix& x, Matrix& y, bool gpu) {
	
	size_t L = layers.size();
	if (L < 1) {
		throw Exception("No layers in this network.");
	}
	
	this->x = &x;
	this->y = &y;
	
	if (L == 1) {
		Layer& single = *layers[0];
		
		single.set_dim(x.n, y.n);
		single.use_gpu(gpu);
		single.init();
		
		return;
	}
	
	Layer& first = *layers[0];
	first.set_dim(x.n, x.n);
	first.use_gpu(false);
	first.init();
	
	for (size_t l = 1; l < L - 1; l++) {
		
		Layer& crt = *layers[l];
		crt.set_dim(x.n, x.n);
		crt.use_gpu(gpu);
		crt.init();
	}
	
	Layer& last = *layers[L - 1];
	last.set_dim(x.n, y.n);
	last.use_gpu(gpu);
	last.init();
}

Matrix& Network::forward() {
	
	size_t L = layers.size();
	if (L < 1) {
		
		throw Exception("No layers in this network.");
	}
	
	Layer& l1 = *layers[0];
	
	Matrix& x = *this->x;
	Matrix& o = l1.foward(x);

	
	Layer& l2 = *layers[1];
	
	CpuMatrix& c = cpu_cast(o);
	CpuMatrix n = CpuMatrix(c.m, c.n);
	
	Matrix& out = l1.foward(o);
	l2.foward(out);
	
	return o;
}

const CpuMatrix Network::cpu_last_grad() const {
	
	size_t L = layers.size();
	
	Layer& last = *layers[L - 1];
	
	CpuMatrix& h = cpu_cast(last.get_fx());
	
	CpuMatrix y = cpu_cast(this->y);
	
	CpuMatrix dg = y - h;
	
	return dg;
}

void Network::backward() {
	
	size_t L = layers.size();
	
	if (gpu) {
		
	} else {
		CpuMatrix dg = cpu_last_grad();
		
		Layer* last = layers[L - 1];
		Matrix* o = &last->backward(dg);
		for (long int i = L - 2; i >= 0; i--) {
			Layer& crt = *layers[i];
			
			o = &crt.backward(*o);
		}
	}
	
}

void Network::update() {
	
	size_t L = layers.size();
	
	for (size_t l = 1; l < L; l++) {
		Layer& crt = *layers[l];
		crt.update(alpha);
	}
	
}

void Network::train(size_t iter) {
	for (size_t i = 0; i < iter; i++) {
		forward();
		backward();
		update();
	}
}

float Network::min_square_error() {
	
	size_t L = layers.size();
	Layer& last = *layers[L - 1];
	if (last.has_fx() == false) {
		//forward();
	}
	
	Matrix& h = last.get_fx();
	return cs::nn::min_square_error(h, *y);
}

Network::~Network() {
	layers.clear();
}

} // namespace math
} // namespace cs

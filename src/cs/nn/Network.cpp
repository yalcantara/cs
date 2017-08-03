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
	l->set_dim(layer.in_dim(), layer.out_dim());
	layers.push_back(l);
}

void Network::operator<<(Sigmoid layer) {
	Sigmoid* l = new Sigmoid();
	l->set_dim(layer.in_dim());
	layers.push_back(l);
}

void Network::set_alpha(float alpha) {
	this->alpha = alpha;
}

float Network::get_alpha() const {
	return this->alpha;
}

void Network::init(Matrix& x, Matrix& y, bool gpu) {
	
	size_t L = layers.size();
	if (L < 1) {
		throw Exception("No layers in this network.");
	}
	
	if (gpu) {
		if (is_gpu(x) == false) {
			throw Exception("Flag gpu is set to TRUE, but the x Matrix is not a GpuMatrix.");
		}
		
		if (is_gpu(y) == false) {
			throw Exception("Flag gpu is set to TRUE, but the y Matrix is not a GpuMatrix.");
		}
	}
	
	this->x = &x;
	this->y = &y;
	this->gpu = gpu;
	
	if (L == 1) {
		Layer& single = *layers[0];
		
		single.use_gpu(gpu);
		single.init();
		
		return;
	}
	
	Layer& first = *layers[0];
	first.use_gpu(gpu);
	first.init();
	
	for (size_t l = 1; l < L - 1; l++) {
		
		Layer& crt = *layers[l];
		crt.use_gpu(gpu);
		crt.init();
	}
	
	Layer& last = *layers[L - 1];
	last.use_gpu(gpu);
	last.init();
}

Matrix& Network::forward() {
	
	size_t L = layers.size();
	if (L < 1) {
		
		throw Exception("No layers in this network.");
	}
	
	Matrix* out = x;
	
	for (size_t l = 0; l < L; l++) {
		Layer& crt = *layers[l];
		
		out = &crt.foward(*out);
	}
	
	return *out;
}

const CpuMatrix Network::cpu_last_grad() const {
	
	size_t L = layers.size();
	
	Layer& last = *layers[L - 1];
	
	CpuMatrix& h = cpu_cast(last.get_fx());
	
	CpuMatrix& y = cpu_cast(this->y);
	
	CpuMatrix dg = h - y;
	
	return dg;
}

const GpuMatrix Network::gpu_last_grad() const {
	
	size_t L = layers.size();
	
	Layer& last = *layers[L - 1];
	
	GpuMatrix& h = gpu_cast(last.get_fx());
	
	GpuMatrix& y = gpu_cast(this->y);
	
	GpuMatrix dg = h - y;
	
	return dg;
}

void Network::backward() {
	
	size_t L = layers.size();
	
	Layer* last = layers[L - 1];
	Matrix* o;
	
	if (gpu) {
		
		GpuMatrix dg = gpu_last_grad();
		
		o = &last->backward(dg);
	} else {
		CpuMatrix dg = cpu_last_grad();
		
		o = &last->backward(dg);
	}
	
	//L - 2 cuz, we already did backward on layers[L - 1]
	for (long int i = L - 2; i >= 0; i--) {
		Layer& crt = *layers[i];
		
		o = &crt.backward(*o);
	}
	
}

void Network::update() {
	
	size_t L = layers.size();
	
	for (size_t l = 0; l < L; l++) {
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
		forward();
	}
	
	Matrix& h = last.get_fx();
	return cs::nn::min_square_error(h, *y);
}

Network::~Network() {
	layers.clear();
}

} // namespace math
} // namespace cs

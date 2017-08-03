/*
 * Sigmoid.cpp
 *
 *  Created on: Mar 6, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/core/lang.h>
#include <cs/nn/Layer.h>
#include <cs/nn/Sigmoid.h>
#include <cs/math/math.h>
#include <cs/nn/gpu_layers.cuh>
#include <stdlib.h>
#include <math.h>

namespace cs {
using namespace core;
using namespace math;
namespace nn {



Sigmoid::Sigmoid() {
	
}

Sigmoid::Sigmoid(size_t dim) {
	set_dim(dim);
}

void Sigmoid::init() {
	//nothing to init
}

void Sigmoid::set_dim(size_t inout){
	Layer::set_dim(inout, inout);
}

Matrix& Sigmoid::foward(const Matrix& x) {
	init_fx(x.m);
	this->x = const_cast<Matrix*>(&x);
	
	x.check_same_dimensions(*fx);
	if (gpu) {
		gpu_foward(gpu_cast(x), gpu_cast(fx));
	} else {
		cpu_foward(cpu_cast(x), cpu_cast(fx));
	}
	
	return *fx;
}

Matrix& Sigmoid::backward(const Matrix& dg) {
	init_dx(x->m, x->n);
	
	dg.check_same_dimensions(*fx);
	if(gpu){
		gpu_backward(gpu_cast(dg));
	}else{
		cpu_backward(cpu_cast(dg));
	}
	
	return *dx;
}

void Sigmoid::update(float alpha) {
	//no need for update
}

void Sigmoid::gpu_foward(const GpuMatrix& x, const GpuMatrix& fx){
	sigmoid_fx(x, fx);
}

void Sigmoid::gpu_backward(const GpuMatrix& dg){
	
	GpuMatrix& x = gpu_cast(this->x);
	GpuMatrix& dx = gpu_cast(this->dx);
	
	sigmoid_dx(x, dx);
	dx.multi(dg);
}

void Sigmoid::cpu_foward(const CpuMatrix& x, const CpuMatrix& fx) {
	
	size_t m = x.m;
	size_t n = x.n;
	
	float* X = x.ptr();
	float* FX = fx.ptr();
	
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			float z = X[i * n + j];
			FX[i * n + j] = cpu_sigmoid_fx(z);
		}
	}
}


void Sigmoid::cpu_backward(const CpuMatrix& dg) {
	
	const CpuMatrix& x = cpu_cast(this->x);
	const CpuMatrix& dx = cpu_cast(this->dx);
	
	size_t m = x.m;
	size_t n = x.n;
	
	float* X = x.ptr();
	float* DX = dx.ptr();
	float* DG = dg.ptr();
	
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			float z = X[i * n + j];
			DX[i * n + j] = cpu_sigmoid_dx(z) * DG[i * n + j];
		}
	}
}

float Sigmoid::cpu_sigmoid_fx(float z) {
	return 1.0f / (1.0f + expf(-z));
}

float Sigmoid::cpu_sigmoid_dx(float z) {
	return cpu_sigmoid_fx(z) * (1 - cpu_sigmoid_fx(z));
}

void Sigmoid::print() const {
	
	println();
	println("Sigmoid");
	println("------------------------------------------------------------------");
	println("In : " + to_string(in));
	println("Out: " + to_string(out));
	println("------------------------------------------------------------------");
	println();
}

Sigmoid::~Sigmoid() {
//nothing to clear
}

} // namespace math
} // namespace cs

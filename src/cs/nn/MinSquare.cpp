/*
 * MinSquare.cpp
 *
 *  Created on: Jun 25, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/math/math.h>
#include <cs/nn/MinSquare.h>
#include <cs/nn/Sigmoid.h>
#include <cs/core/lang.h>

namespace cs {
using namespace math;
using namespace core;
namespace nn {

MinSquare::MinSquare() {
	// TODO Auto-generated constructor stub
	
}

void MinSquare::init() {
	//nothing to init
}

void MinSquare::set_dim(size_t inout){
	Layer::set_dim(inout, inout);
}

Matrix& MinSquare::foward(const Matrix& x) {
	init_fx(x.m);
	this->x = const_cast<Matrix*>(&x);
	
	if (gpu) {
		gpu_foward(gpu_cast(x), gpu_cast(fx));
	} else {
		cpu_foward(cpu_cast(x), cpu_cast(fx));
	}
	
	return *fx;
}

Matrix& MinSquare::backward(const Matrix& dg) {
	init_dx(x->m, x->n);
	
	if(gpu){
		gpu_backward(gpu_cast(dg));
	}else{
		cpu_backward(cpu_cast(dg));
	}
	
	return *dx;
}

void MinSquare::update(float alpha) {
	//no need for update
}

void MinSquare::gpu_foward(const GpuMatrix& x, const GpuMatrix& fx){
	
}

void MinSquare::gpu_backward(const GpuMatrix& dg){
	
	GpuMatrix& x = gpu_cast(this->x);
	GpuMatrix& dx = gpu_cast(this->dx);
	
	
}

void MinSquare::cpu_foward(const CpuMatrix& x, const CpuMatrix& fx) {
	
	size_t m = x.m;
	size_t n = x.n;
	
	float* X = x.ptr();
	float* FX = fx.ptr();
	
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			FX[i * n + j] = X[i * n + j];
		}
	}
}


void MinSquare::cpu_backward(const CpuMatrix& y) {
	
	const CpuMatrix& fx = cpu_cast(this->fx);
	const CpuMatrix& dx = cpu_cast(this->dx);
	
	size_t m = fx.m;
	size_t n = fx.n;
	
	
	float* Y = y.ptr();
	
	float* FX = fx.ptr();
	float* DX = dx.ptr();
	
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			DX[i * n + j] = FX[i * n + j] - Y[i * n + j];
		}
	}
}


void MinSquare::print() const {
	
	println();
	println("MinSquare Loss");
	println("------------------------------------------------------------------");
	println("In : " + to_string(in));
	println("Out: " + to_string(out));
	println("------------------------------------------------------------------");
	println();
}


MinSquare::~MinSquare() {
	// TODO Auto-generated destructor stub
}

} // namespace math 
} // namespace cs 

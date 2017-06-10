/*
 * Affine.cpp
 *
 *  Created on: Feb 19, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/core/lang.h>
#include <cs/math/CpuVector.h>
#include <cs/math/GpuVector.h>
#include <cs/math/math.h>
#include <cs/nn/Affine.h>
#include <stddef.h>

#include <cs/nn/gpu_layers.cuh>

namespace cs {
using namespace core;
using namespace math;
namespace nn {

Affine::Affine() :
		Layer() {
}

void Affine::release() {
	if (w) {
		delete w;
	}
	
	if (b) {
		delete b;
	}
	
	if (db) {
		delete db;
	}
	
	if (dw) {
		delete dw;
	}
}

void Affine::init() {
	release();
	if (gpu) {
		w = new GpuMatrix(in, out);
		w->randn();
		
		b = new GpuVector(out);
		b->randn();
		
		dw = new GpuMatrix(in, out);
		db = new GpuVector(out);
	} else {
		w = new CpuMatrix(in, out);
		w->randn();
		
		b = new CpuVector(out);
		b->randn();
		
		dw = new CpuMatrix(in, out);
		db = new CpuVector(out);
	}
}



void Affine::set_weights(const Matrix& weights) {
	weights.copy(*w);
}

void Affine::set_bias(const Vector& bias) {
	bias.copy(*b);
}




Matrix& Affine::foward(const Matrix& x) {
	init_fx(x.m);
	this->x = const_cast<Matrix*>(&x);
	
	x.affine(*w, *b, *fx);
	return *fx;
}

Matrix& Affine::backward(const Matrix& dg) {
	init_dx(x->m, x->n);
	
	if (gpu) {
		return gpu_backward(gpu_cast(dg));
	}
	
	return cpu_backward(cpu_cast(dg));
}

Matrix& Affine::gpu_backward(const GpuMatrix& dg) {
	
	GpuMatrix& x = gpu_cast(this->x);
	GpuMatrix& w = gpu_cast(this->w);
	
	GpuMatrix& dx = gpu_cast(this->dx);
	GpuMatrix& dw = gpu_cast(this->dw);
	GpuVector& db = gpu_cast(this->db);
	
	affine_dx(x, w, dg, dx, dw, db);
	return dx;
}

Matrix& Affine::cpu_backward(const CpuMatrix& dg) {
	
	CpuMatrix& x = cpu_cast(this->x);
	CpuMatrix& w = cpu_cast(this->w);
	
	CpuMatrix& dx = cpu_cast(this->dx);
	CpuMatrix& dw = cpu_cast(this->dw);
	CpuVector& db = cpu_cast(this->db);
	
	dx.clear();
	dw.clear();
	db.clear();
	
	size_t m = x.m;
	size_t n = x.n;
	size_t p = w.n;
	
	float* X = x.ptr();
	float* W = w.ptr();
	
	float* DG = dg.ptr();
	float* DX = dx.ptr();
	float* DW = dw.ptr();
	float* DB = db.ptr();
	
	for (int i = 0; i < m; i++) {
		for (int k = 0; k < p; k++) {
			
			float fdg = DG[i * p + k];
			for (int j = 0; j < n; j++) {
				float dfw = fdg * W[j * p + k];
				float dfx = fdg * X[i * n + j];
				
				DX[i * n + j] += dfw;
				DW[j * p + k] += dfx;
			}
			
			DB[k] += fdg;
		}
	}
	
	return dx;
}

void Affine::update(float alpha) {
	
	if (gpu) {
		gpu_update(alpha);
	} else {
		cpu_update(alpha);
	}
}

void Affine::gpu_update(float alpha) {
	
	GpuMatrix& w = gpu_cast(this->w);
	GpuVector& b = gpu_cast(this->b);
	
	GpuMatrix& dw = gpu_cast(this->dw);
	GpuVector& db = gpu_cast(this->db);
	
	size_t m = x->n;
	float scalar = -alpha / m;
	
	update_params(w, dw, scalar);
	update_params(b, db, scalar);
	
}

void Affine::cpu_update(float alpha) {
	
	CpuMatrix& w = cpu_cast(this->w);
	CpuVector& b = cpu_cast(this->b);
	
	CpuMatrix& dw = cpu_cast(this->dw);
	CpuVector& db = cpu_cast(this->db);
	
	float* W = w.ptr();
	float* B = b.ptr();
	
	float* DW = dw.ptr();
	float* DB = db.ptr();
	
	size_t m = x->n;
	size_t length = in * out;
	
	//for efficiency, instead of using Wj := Wj - (alpha/m)*DWj
	//we pre-compute (alpha/m) which is just a constant.
	
	float scalar = -alpha / m;
	for (int j = 0; j < length; j++) {
		W[j] += scalar * DW[j];
	}
	
	for (int j = 0; j < out; j++) {
		B[j] += scalar * DB[j];
	}
}

void Affine::print() const {
	
	println();
	println("Affine");
	println("------------------------------------------------------------------");
	println("Weights:");
	w->print();
	
	println("Bias:");
	b->print();
	
	println("DW:");
	dw->print();
	
	println("DB:");
	db->print();
	println("------------------------------------------------------------------");
	println();
	
}

Affine::~Affine() {
	release();
}

} // namespace nn
} // namespace cs

/*
 * Affine.cpp
 *
 *  Created on: Feb 19, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/math/CpuMatrix.h>
#include <cs/math/CpuVector.h>
#include <cs/math/GpuMatrix.h>
#include <cs/math/GpuVector.h>
#include <cs/math/math.h>
#include <cs/nn/Affine.h>
#include <stddef.h>

namespace cs {
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
	
	if (fx) {
		delete fx;
	}
	
	if (dx) {
		delete dx;
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
		
		CpuMatrix m = {{0, 1}, {0, 2}};
		
		float* aa = m.ptr();
		float* bb = dynamic_cast<CpuMatrix&>(*w).ptr();
		
		for(int i =0; i < m.length; i++){
			bb[i] = aa[i];
		}
		
		
		b = new CpuVector(out);
		//b->randn();
		
		dw = new CpuMatrix(in, out);
		db = new CpuVector(out);
	}
}

void Affine::init_fx(size_t m) {
	//If we already have created a fx then, we have to check if the dimensions 
	//are right for the incoming x. If it passes that test, then no need to create 
	//another one. If the dimensions does not match, we have to delete the 
	//previous and create a new one.
	if (fx) {
		if (fx->m == m && fx->n == out) {
			//it's already created and the dimensions are fine
			return;
		}
		
		delete fx;
		if (gpu) {
			fx = new GpuMatrix(m, out);
		} else {
			fx = new CpuMatrix(m, out);
		}
	} else {
		if (gpu) {
			fx = new GpuMatrix(m, out);
		} else {
			fx = new CpuMatrix(m, out);
		}
	}
}

void Affine::init_dx(size_t m, size_t n) {
	//If we already have created a dx then, we have to check if the dimensions 
	//are right for the incoming x. If it passes that test, then no need to create 
	//another one. If the dimensions does not match, we have to delete the 
	//previous and create a new one.
	if (dx) {
		if (dx->m == m && dx->n == n) {
			//it's already created and the dimensions are fine
			return;
		}
		
		delete dx;
		if (gpu) {
			dx = new GpuMatrix(m, n);
		} else {
			dx = new CpuMatrix(m, n);
		}
	} else {
		if (gpu) {
			dx = new GpuMatrix(m, n);
		} else {
			dx = new CpuMatrix(m, n);
		}
	}
}

Matrix& Affine::foward(Matrix& x) {
	this->x = &x;
	init_fx(x.m);
	x.affine(w, b, fx);
	return *fx;
}

Matrix& Affine::backward(Matrix& dg){
	if(gpu){
		
	}
	
	init_dx(x->m, x->n);
	
	dx->clear();
	dw->clear();
	db->clear();
	
	return cpu_backward(dg);
}

Matrix& Affine::cpu_backward(Matrix& dg) {
	
	CpuMatrix& cdg = cpu_cast(dg);
	
	CpuMatrix& x = cpu_cast(this->x);
	CpuMatrix& w = cpu_cast(this->w);
	
	CpuMatrix& dx = cpu_cast(this->dx);
	CpuMatrix& dw = cpu_cast(this->dw);
	CpuVector& db = cpu_cast(this->db);
	
	size_t m = x.m;
	size_t n = x.n;
	size_t p = w.n;
	
	init_dx(m, n);
	
	float* DG = cdg.ptr();
	
	float* X = x.ptr();
	float* W = w.ptr();
	
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
	
	CpuMatrix& w = cpu_cast(this->w);
	CpuVector& b = cpu_cast(this->b);
	
	CpuMatrix& dw = cpu_cast(this->dw);
	CpuVector& db = cpu_cast(this->db);
	
	float* W = w.ptr();
	float* B = w.ptr();
	
	float* DW = db.ptr();
	float* DB = db.ptr();
	
	size_t length = in * out;
	for (int i = 0; i < length; i++) {
		W[i] -= DW[i] * alpha;
	}
	
	for (int j = 0; j < out; j++) {
		B[j] -= -DB[j] * alpha;
	}
}

Affine::~Affine() {
	release();
}

} // namespace nn
} // namespace cs

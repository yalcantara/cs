/*
 * Affine.cpp
 *
 *  Created on: Feb 19, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/core/lang.h>
#include <cs/math/CpuMatrix.h>
#include <cs/math/CpuVector.h>
#include <cs/math/GpuMatrix.h>
#include <cs/math/GpuVector.h>
#include <cs/nn/Affine.h>
#include <stdlib.h>

namespace cs {
using namespace core;
using namespace math;
namespace nn {

Affine::Affine() :
		Layer() {
}

void Affine::init() {
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
		db = new GpuVector(out);
	}
}

Matrix& Affine::foward(Matrix& x) {
	this->x = &x;
	x.affine(w, b, fx);
	return *fx;
}

Affine::~Affine() {
	if (w) {
		delete (w);
	}
}

} // namespace nn
} // namespace cs

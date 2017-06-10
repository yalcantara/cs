/*
 * Layer.cpp
 *
 *  Created on: Feb 19, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/core/lang.h>
#include <cs/math/CpuMatrix.h>
#include <cs/math/GpuMatrix.h>
#include <cs/nn/Layer.h>
#include <stddef.h>

namespace cs {
using namespace core;
namespace nn {

Layer::Layer() {
	// TODO Auto-generated constructor stub
	
}

void Layer::use_gpu(bool val){
	gpu = val;
}

void Layer::set_dim(size_t input, size_t output){
	in = input;
	out = output;
}

Matrix& Layer::get_dx()const{
	check_null(dx);
	return *dx;
}


void Layer::init_fx(size_t m) {
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

void Layer::init_dx(size_t m, size_t n) {
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
	}
	
	if (gpu) {
		dx = new GpuMatrix(m, n);
	} else {
		dx = new CpuMatrix(m, n);
	}
}

Layer::~Layer() {
	if(fx){
		delete fx;
	}
	
	if(dx){
		delete dx;
	}
}

} // namespace nn
} // namespace cs 

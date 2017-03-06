/*
 * Layer.cpp
 *
 *  Created on: Feb 19, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/nn/Layer.h>
#include <cs/core/lang.h>

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

Layer::~Layer() {
}

} // namespace nn
} // namespace cs 

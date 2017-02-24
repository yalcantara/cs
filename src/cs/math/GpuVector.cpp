/*
 * GpuVector.cpp
 *
 *  Created on: Jan 28, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/core/Exception.h>
#include <cs/gpu/gpu.h>
#include <cs/math/GpuVector.h>
#include <cs/math/math.h>

namespace cs {

using namespace gpu;
using namespace core;

namespace math {

GpuVector::GpuVector(size_t length) :
		GpuVector(length, true) {
}

GpuVector::GpuVector(size_t length, bool clear) :
		length(length) {
	devPtr = gpu_malloc(length, clear);
}

GpuVector::GpuVector(const CpuVector& other) :
		GpuVector(other.length, false) {
	float* src = other.ptr();
	copy_cpu_to_gpu(src, devPtr, length);
}

GpuVector::GpuVector(const GpuVector& other) :
		GpuVector(other.length, false) {
	float* src = other.ptr();
	gpu::copy_gpu_to_gpu(src, devPtr, length);
}

GpuVector::GpuVector(const initializer_list<float> &list) :
		GpuVector(list.size(), false) {
	
	CpuVector temp = CpuVector(list);
	float* src = temp.ptr();
	copy_cpu_to_gpu(src, devPtr, length);
}

GpuVector& GpuVector::operator=(const GpuVector& other) {
	
	if (&other == this) {
		return *this;
	}

	//There are 2 cases:
	//a - the destination is not initialized
	//b - the dimensions are equals (if not, throw an exception)
	
	if (devPtr == nullptr) {
		
		//allocate and copy
		const_cast<size_t&>(length) = other.length;
		devPtr = gpu_malloc(other.length, false);
		copy_gpu_to_gpu(other.devPtr, devPtr, length);
	} else {
		check_same_length(other);
		//the vector is already initialized
		//just copy
		copy_gpu_to_gpu(other.devPtr, devPtr, length);
	} 
	
	return *this;
}

void GpuVector::check_same_length(const GpuVector& b) const {
	if (b.length != length) {
		throw Exception(
				"The length must be the same. Expected " + to_string(length) + ", but got: "
						+ to_string(b.length) + " instead.");
	}
}

const GpuVector GpuVector::operator+(const GpuVector& b) const{
	check_same_length(b);
	
	GpuVector ans = GpuVector(length, false);
	
	gpu_add(devPtr, b.devPtr, ans.devPtr, length);
	
	return ans;
}

const GpuVector GpuVector::operator-(const GpuVector& b) const{
	check_same_length(b);
	
	GpuVector ans = GpuVector(length, false);
	
	gpu_sub(devPtr, b.devPtr, ans.devPtr, length);
	
	return ans;
}

const GpuVector GpuVector::operator-() const{
	return (*this) * -1;
}

const GpuVector GpuVector::operator*(const GpuVector& b) const{
	check_same_length(b);
	GpuVector ans = GpuVector(length, false);
	
	gpu_mult(devPtr, b.devPtr, ans.devPtr, length);
	
	return ans;
}

const GpuVector GpuVector::operator*(const float scalar) const{
	
	GpuVector ans = GpuVector(length, false);
	
	gpu_mult(devPtr, scalar, ans.devPtr, length);
	
	return ans;
}

const GpuVector GpuVector::operator/(const GpuVector& b) const{
	check_same_length(b);
	GpuVector ans = GpuVector(length, false);
	
	gpu_div(devPtr, b.devPtr, ans.devPtr, length);
	
	return ans;
}

const GpuVector GpuVector::operator/(const float scalar) const{
	
	GpuVector ans = GpuVector(length, false);
	float div = 1.0/scalar;
	gpu_mult(devPtr, div, ans.devPtr, length);
	
	return ans;
}

const GpuVector GpuVector::operator^(const float exp) const{
	
	GpuVector ans = GpuVector(length, false);
	
	gpu_pow(devPtr, exp, ans.devPtr, length);
	
	return ans;
}



void GpuVector::addi(const GpuVector& b){
	check_same_length(b);
	gpu_add_inplace(devPtr, b.devPtr, length);
}
void GpuVector::subi(const GpuVector& b){
	check_same_length(b);
	gpu_sub_inplace(devPtr, b.devPtr, length);
}
void GpuVector::multi(const GpuVector& b){
	check_same_length(b);
	gpu_mult_inplace(devPtr, b.devPtr, length);
}
void GpuVector::multi(const float scalar){
	gpu_mult_inplace(devPtr, scalar, length);
}
void GpuVector::divi(const GpuVector& b){
	check_same_length(b);
	gpu_div_inplace(devPtr, b.devPtr, length);
}
void GpuVector::divi(const float scalar){
	//inverted scalar
	const float div = 1.0 / scalar;
	gpu_mult_inplace(devPtr, div, length);
}
void GpuVector::powi(const float exp){
	gpu_pow_inplace(devPtr, exp, length);
}


float GpuVector::sum() const {
	
	float ans = gpu_sum(devPtr, length);
	
	return ans;
}


const CpuVector GpuVector::cpu() const {
	
	CpuVector ans = CpuVector(length, false);
	
	float* dest = ans.ptr();
	copy_gpu_to_cpu(devPtr, dest, length);
	
	return ans;
}

void GpuVector::randn(){
	CpuVector tmp = cs::math::randn(length);
	
	float* src = tmp.ptr();
	copy_cpu_to_gpu(src, devPtr, length);
}

float* GpuVector::ptr() const {
	return devPtr;
}

void GpuVector::print() const {
	CpuVector temp = cpu();
	temp.print();
}

GpuVector::~GpuVector() {
	if (devPtr) {
		gpu_free(devPtr);
	}
}
}	//namespace math
}	//namespace cs

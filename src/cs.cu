/*
 ============================================================================
 Name        : cs.cu
 Author      : Yaison Alcantara Alcantara
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <cs/core/Exception.h>
#include <cs/core/lang.h>
#include <cs/math/CpuMatrix.h>
#include <cs/math/GpuMatrix.h>
#include <cs/math/GpuVector.h>
#include <cs/math/math.h>
#include <cs/nn/Affine.h>
#include <cs/nn/errors.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <cs/nn/gpu_layers.cuh>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;
using namespace cs::core;
using namespace cs::math;
using namespace cs::nn;

void performance() {
	
	size_t d = 1000;
	GpuMatrix a = randn(d, d);
	GpuMatrix b = randn(d, d);
	
	CpuMatrix ac = randn(d, d);
	CpuMatrix bc = randn(d, d);
	
	for (int i = 0; i < 5; i++) {
		time_t now = clock();
		auto c = a.dot(b);
		auto cpu = c.cpu();
		double took = clock() - now;
		double millis = took / CLOCKS_PER_SEC * 1000.0;
		printf("millis: %8d\n", (int) millis);
	}
	
	println();
	for (int i = 0; i < 5; i++) {
		time_t now = clock();
		auto cc = ac.dot(bc);
		double took = clock() - now;
		double millis = took / CLOCKS_PER_SEC * 1000.0;
		printf("millis: %8d\n", (int) millis);
	}
}

void test1() {
	GpuMatrix a = { { 1, 2, 3 }, { 4, 5, 6 } };
	GpuMatrix b = { { 1, 2 }, { 3, 4 }, { 5, 6 } };
	
	auto scal = 2 * a;
	scal.print();
	
	auto d = a.dot(b);
	d.print();
	GpuVector v = { 1, 2 };
	auto c = a.affine(b, v);
	
	c.print();
}

void test2() {
	try {
		srand(time(NULL));
		
		Affine f = Affine();
		f.use_gpu(false);
		CpuMatrix x = { { 0 }, { 1 } };
		
		CpuMatrix y = { { 1 }, { 0 } };
		
		println("X:");
		x.print();
		
		println("Y");
		y.print();
		
		f.set_dim(x.n, y.n);
		f.init();
		CpuMatrix w = { { 0 } };
		CpuVector b = { 2 };
		f.set_weights(w);
		f.set_bias(b);
		
		float j;
		float alpha = 0.1;
		
		//f.print();
		int iter = 3;
		for (int i = 0; i <= iter; i++) {
			println("====================================");
			//f.print();
			Matrix& h = f.foward(x);
			if (iter < 10 || i % (iter / 10) == 0) {
				j = min_square_error(h, y);
				println("j: " + to_string(j));
			}
			
			h.print();
			f.print();
			CpuMatrix dg = cpu_cast(h) - y;
			
			f.backward(dg);
			
			f.update(alpha);
			
		}
		
		println("ended");
	} catch (Exception& ex) {
		println("Exception thrown");
		println(ex.what());
	}
}

void gpu_test() {
	try {
		srand(time(NULL));
		
		Affine f = Affine();
		f.use_gpu(true);
		GpuMatrix x = { { 0 }, { 1 } };
		
		GpuMatrix y = { { 1 }, { 0 } };
		
		println("X:");
		x.print();
		
		println("Y");
		y.print();
		
		f.set_dim(x.n, y.n);
		f.init();
		GpuMatrix w = { { 0 } };
		GpuVector b = { 2 };
		f.set_weights(w);
		f.set_bias(b);
		
		float j;
		float alpha = 0.1;
		
		//f.print();
		int iter = 100;
		for (int i = 0; i <= iter; i++) {
			//println("===================================");
			//f.print();
			Matrix& h = f.foward(x);
			
			if (iter <= 10 || i % (iter / 10) == 0) {
				j = min_square_error(h, y);
				//println("=======================================");
				println("j: " + to_string(j));
			}
			
			GpuMatrix dg = gpu_cast(h) - y;
			
			f.backward(dg);
			f.update(alpha);
		}
		
		Matrix& h = f.foward(x);
		
		h.print();
		println("ended");
	} catch (Exception& ex) {
		println("Exception thrown");
		println(ex.what());
	}
}

cublasHandle_t cublas_handle = nullptr;

const char* _cuda_get_error_enum(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";
		
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
		
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
		
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
		
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
		
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
		
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
		
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	
	return "<unknown>";
}

void check_cublas(cublasStatus_t status) {
	if (status == CUBLAS_STATUS_SUCCESS) {
		return;
	}
	
	fprintf(stderr, "CUBLAS error %d\nMessage: %s.\n", status, _cuda_get_error_enum(status));
	fflush(stderr);
	throw Exception("Cuda error");
}

int main(void) {
	gpu_test();
	//test2();
	

	
	return 0;
}


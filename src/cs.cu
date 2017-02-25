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
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

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

int main(void) {
	try {
		srand(time(NULL));
		
		Affine* l = new Affine();
		l->use_gpu(false);
		
		CpuMatrix x = { { 1, 2 }, { 3, 4 } };
		CpuMatrix y = { { 0 }, { 1 } };
		
		Matrix& base = x;
		base.print();
		
		l->set_dim(x.n, 1);
		l->init();
		
		
		println("klk");
	} catch (Exception& ex) {
		println("Exception thrown");
		println(ex.what());
	}
	return 0;
}


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
#include <cs/core/utils.h>
#include <cs/data/Grid.h>
#include <cs/data/GridInfo.h>
#include <cs/gpu/gpu.h>
#include <cs/math/CpuMatrix.h>
#include <cs/math/CpuVector.h>
#include <cs/math/GpuMatrix.h>
#include <cs/math/GpuVector.h>
#include <cs/math/math.h>
#include <cs/nn/Affine.h>
#include <cs/nn/errors.h>
#include <cs/nn/MinSquare.h>
#include <cs/nn/Network.h>
#include <cs/nn/Sigmoid.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <string>

using namespace std;
using namespace cs::core;
using namespace cs::math;
using namespace cs::nn;
using namespace cs::gpu;
using namespace cs::data;

void memtest() {
	//test that the memory is ok by doing some memory allocation/dealocation etc...
	
	float* test = (float*) malloc(sizeof(float) * 5);
	free(test);
}

void hit(CpuMatrix& h, CpuMatrix& y) {
	
	size_t total = y.m;
	size_t classes = y.n;
	size_t good = 0;
	
	for (size_t i = 0; i < total; i++) {
		for (size_t j = 0; j < classes; j++) {
			float val = y.get(i, j);
			float ans = h.get(i, j);
			
			if (val == 1) {
				if (ans >= 0.5) {
					good++;
				}
			} else {
				if (ans < 0.5) {
					good++;
				}
			}
		}
	}
	
	println("======================================================");
	printf("total: %6d\n", (int) total);
	printf("good : %6d\n", (int) good);
	printf("perc : %6.0f%\n", (float) (good * 100.0 / (total * classes)));
	println("======================================================");
}

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
		int iter = 100;
		for (int i = 0; i <= iter; i++) {
			//f.print();
			Matrix& h = f.foward(x);
			if (iter < 10 || i % (iter / 10) == 0) {
				j = min_square_error(h, y);
				printf("iter: %6d,  j: %12.8f\n", i, j);
			}
			
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
		GpuMatrix x = { { 0, 0 }, { 0, 1 } };
		
		GpuMatrix y = { { 1 }, { 0 } };
		
		println("X:");
		x.print();
		
		println("Y");
		y.print();
		
		f.set_dim(x.n, y.n);
		f.init();
		println("Affine initialized");
		GpuMatrix w = { { -1 }, { 5 } };
		
		GpuVector b = { 2 };
		f.set_weights(w);
		println("Weights set");
		f.set_bias(b);
		println("Bias set");
		float j;
		float alpha = 0.1;
		
		f.print();
		println("About to train");
		int iter = 1000;
		for (int i = 0; i <= iter; i++) {
			//println("===================================");
			//f.print();
			
			Matrix& h = f.foward(x);
			
			if (iter <= 10 || i % (iter / 10) == 0) {
				j = min_square_error(h, y);
				//println("=======================================");
				printf("iter: %6d  J: %12.8f", i, j);
				println();
			}
			
			GpuMatrix dg = gpu_cast(h) - y;
			
			f.backward(dg);
			
			f.update(alpha);
		}
		
		println();
		Matrix& h = f.foward(x);
		
		f.print();
		h.print();
		println("ended");
	} catch (Exception& ex) {
		println("Exception thrown");
		println(ex.what());
	}
}

void trans_test1() {
	
	println("===================================================");
	println("A^T x B  case 1");
	//continue here: test sigmoid
	GpuMatrix a = { { 1, 2 }, { 3, 4 } };
	GpuMatrix b = { { 1, 0, 0 }, { 3, 1, 1 } };
	
	GpuMatrix c = GpuMatrix(a.n, b.n);
	
	size_t m = a.m;
	size_t n = a.n;
	size_t p = b.n;
	float* A = a.ptr();
	float* B = b.ptr();
	float* C = c.ptr();
	
	gpu_dot(A, true, B, C, m, n, p);
	
	println("The ans should be:");
	GpuMatrix ans = { { 10, 3, 3 }, { 14, 4, 4 } };
	ans.print();
	
	println("Got:");
	c.print();
}

void trans_test2() {
	println("===================================================");
	println("A^T x B  case 2");
	//continue here: test sigmoid
	GpuMatrix a = { { 1, 2, 0 }, { 1, 0, 1 } };
	GpuMatrix b = { { 1, 2, 3, 0 }, { 2, 0, 3, 5 } };
	
	GpuMatrix c = GpuMatrix(a.n, b.n);
	
	size_t m = a.m;
	size_t n = a.n;
	size_t p = b.n;
	float* A = a.ptr();
	float* B = b.ptr();
	float* C = c.ptr();
	
	gpu_dot(A, true, B, C, m, n, p);
	
	println("The ans should be:");
	GpuMatrix ans = { { 3, 2, 6, 5 }, { 2, 4, 6, 0 }, { 2, 0, 3, 5 } };
	ans.print();
	
	println("Got:");
	c.print();
}

void trans_test3() {
	println("===================================================");
	println("A x B^T  case 1");
	//continue here: test sigmoid
	GpuMatrix a = { { 1, 2 }, { 3, 4 } };
	GpuMatrix b = { { 1, 0 }, { 0, 3 }, { 1, 1 } };
	
	GpuMatrix c = GpuMatrix(a.m, b.m);
	
	size_t m = a.m;
	size_t n = a.n;
	size_t o = b.m;
	size_t p = b.n;
	float* A = a.ptr();
	float* B = b.ptr();
	float* C = c.ptr();
	
	b.print();
	gpu_dot(A, B, true, C, m, n, o, p);
	
	println("The ans should be:");
	GpuMatrix ans = { { 1, 6, 3 }, { 3, 12, 7 } };
	ans.print();
	
	println("Got:");
	c.print();
}

void trans_test4() {
	println("===================================================");
	println("A x B^T  case 2");
	//continue here: test sigmoid
	GpuMatrix a = { { 1, 2 }, { 1, 0 }, { 4, 5 } };
	GpuMatrix b = { { 1, 2 }, { 3, 0 } };
	
	GpuMatrix c = GpuMatrix(a.m, b.m);
	
	size_t m = a.m;
	size_t n = a.n;
	size_t o = b.m;
	size_t p = b.n;
	float* A = a.ptr();
	float* B = b.ptr();
	float* C = c.ptr();
	
	gpu_dot(A, B, true, C, m, n, o, p);
	
	println("The ans should be:");
	GpuMatrix ans = { { 5, 3 }, { 1, 3 }, { 14, 12 } };
	ans.print();
	
	println("Got:");
	c.print();
}

void sigmoid_test() {
	
	GpuMatrix a = { { 1, -1 }, { -16, 16 }, { -100, 100 }, { 0, 0 } };
	
	GpuMatrix b = GpuMatrix(a.m, a.n);
	
	a.print();
	Sigmoid s = Sigmoid();
	s.use_gpu(true);
	s.set_dim(a.n);
	
	Matrix& ans = s.foward(a);
	
	ans.print();
}

void sigmoid_test2() {
	try {
		srand(time(NULL));
		
		GpuMatrix x = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		GpuMatrix y = { { 1 }, { 0 }, { 0 }, { 1 } };
		
		println("X:");
		x.print();
		
		println("Y");
		y.print();
		
		Affine f = Affine();
		f.use_gpu(true);
		f.set_dim(x.n, y.n);
		f.init();
		
		Sigmoid s = Sigmoid();
		s.use_gpu(true);
		s.set_dim(f.out_dim());
		s.init();
		
		println("Bias set");
		float j;
		float alpha = 0.1;
		
		f.print();
		println("About to train");
		int iter = 10000;
		for (int i = 0; i <= iter; i++) {
			//println("===================================");
			//f.print();
			
			Matrix& h1 = f.foward(x);
			Matrix& h2 = s.foward(h1);
			
			if (iter <= 10 || i % (iter / 10) == 0) {
				j = min_square_error(h2, y);
				//println("=======================================");
				printf("iter: %6d  J: %12.8f", i, j);
				println();
			}
			
			GpuMatrix dg = gpu_cast(h2) - y;
			Matrix& b = s.backward(dg);
			f.backward(b);
			
			f.update(alpha);
		}
		
		println();
		Matrix& h1 = f.foward(x);
		Matrix& h2 = s.foward(h1);
		println("Ans:");
		h2.print();
		println("ended");
	} catch (Exception& ex) {
		println("Exception thrown");
		println(ex.what());
	}
}

void sigmoid_test3() {
	try {
		srand(time(NULL));
		
		CpuMatrix x = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		CpuMatrix y = { { 1 }, { 0 }, { 0 }, { 1 } };
		
		println("X:");
		x.print();
		
		println("Y");
		y.print();
		
		bool gpu = false;
		
		Affine f1 = Affine();
		f1.use_gpu(gpu);
		f1.set_dim(x.n, x.n);
		f1.init();
		
		Sigmoid s1 = Sigmoid();
		s1.use_gpu(gpu);
		s1.set_dim(f1.out_dim());
		s1.init();
		
		Affine f2 = Affine();
		f2.use_gpu(gpu);
		f2.set_dim(x.n, y.n);
		f2.init();
		
		Sigmoid s2 = Sigmoid();
		s2.use_gpu(gpu);
		s2.set_dim(f2.out_dim());
		s2.init();
		
		println("Bias set");
		float j;
		float alpha = 0.1;
		
		println("About to train");
		int iter = 100000;
		for (int i = 0; i <= iter; i++) {
			//println("===================================");
			//f.print();
			
			Matrix& h1 = f1.foward(x);
			Matrix& h2 = s1.foward(h1);
			Matrix& h3 = f2.foward(h2);
			Matrix& h4 = s2.foward(h3);
			
			if (iter <= 10 || i % (iter / 10) == 0) {
				j = min_square_error(h4, y);
				//println("=======================================");
				printf("iter: %8d  J: %12.8f", i, j);
				println();
			}
			
			CpuMatrix dg = cpu_cast(h4) - y;
			Matrix& b1 = s2.backward(dg);
			Matrix& b2 = f2.backward(b1);
			Matrix& b3 = s1.backward(b2);
			f1.backward(b3);
			
			f2.update(alpha);
			f1.update(alpha);
		}
		
		println();
		
		Matrix& h1 = f1.foward(x);
		Matrix& h2 = s1.foward(h1);
		Matrix& h3 = f2.foward(h2);
		Matrix& h4 = s2.foward(h3);
		
		println("Ans:");
		h4.print();
		println("ended");
	} catch (Exception& ex) {
		println("Exception thrown");
		println(ex.what());
	}
}

void iris_test_gpu() {
	
	string data = ffull("files/iris.data");
	
	GridInfo info = GridInfo(4);
	
	Grid g = Grid(data);
	//g.shuffle();
	
	GpuMatrix x = g.toMatrix(0, 4, true);
	
	GpuMatrix y = g.toMatrix(4, 5, false);
	
	bool gpu = true;
	
	Affine f1 = Affine();
	f1.use_gpu(gpu);
	f1.set_dim(x.n, x.n);
	f1.init();
	
	Sigmoid s1 = Sigmoid();
	s1.use_gpu(gpu);
	s1.set_dim(f1.out_dim());
	s1.init();
	
	Affine f2 = Affine();
	f2.use_gpu(gpu);
	f2.set_dim(s1.out_dim(), y.n);
	f2.init();
	
	Sigmoid s2 = Sigmoid();
	s2.use_gpu(gpu);
	s2.set_dim(f2.out_dim());
	s2.init();
	
	println("About to train");
	int iter = 100000;
	float j;
	float alpha = 0.1;
	for (int i = 0; i <= iter; i++) {
		//println("===================================");
		//f.print();
		
		Matrix& h1 = f1.foward(x);
		Matrix& h2 = s1.foward(h1);
		Matrix& h3 = f2.foward(h2);
		Matrix& h4 = s2.foward(h3);
		
		if (iter <= 10 || i % (iter / 10) == 0) {
			j = min_square_error(h4, y);
			//println("=======================================");
			printf("iter: %8d  J: %12.8f", i, j);
			println();
		}
		
		GpuMatrix dg = gpu_cast(h4) - y;
		Matrix& b1 = s2.backward(dg);
		Matrix& b2 = f2.backward(b1);
		Matrix& b3 = s1.backward(b2);
		f1.backward(b3);
		
		f2.update(alpha);
		f1.update(alpha);
	}
	
	println();
	Matrix& h1 = f1.foward(x);
	Matrix& h2 = s1.foward(h1);
	Matrix& h3 = f2.foward(h2);
	Matrix& h4 = s2.foward(h3);
	
	println("Ans:");
	h4.print();
	println("ended");
}

void test_data() {
	
	string data = ffull("files/iris.data");
	
	Grid g = Grid(data);
	//g.shuffle();
	
	CpuMatrix x = g.toMatrix(0, 4, false);
	
	CpuMatrix y = g.toMatrix(4, 5, false);
	
	bool gpu = false;
	
	Affine f1 = Affine();
	f1.use_gpu(gpu);
	f1.set_dim(x.n, x.n);
	f1.init();
	
	Sigmoid s1 = Sigmoid();
	s1.use_gpu(gpu);
	s1.set_dim(f1.out_dim());
	s1.init();
	
	Affine f2 = Affine();
	f2.use_gpu(gpu);
	f2.set_dim(s1.out_dim(), y.n);
	f2.init();
	
	Sigmoid s2 = Sigmoid();
	s2.use_gpu(gpu);
	s2.set_dim(f2.out_dim());
	s2.init();
	
	MinSquare ms = MinSquare();
	ms.set_dim(s2.out_dim());
	ms.init();
	
	println("About to train");
	int iter = 100000;
	float j;
	float alpha = 0.1;
	for (int i = 0; i <= iter; i++) {
		//println("===================================");
		//f.print();
		
		Matrix& h1 = f1.foward(x);
		Matrix& h2 = s1.foward(h1);
		Matrix& h3 = f2.foward(h2);
		Matrix& h4 = s2.foward(h3);
		Matrix& h5 = ms.foward(h4);
		
		if (iter <= 10 || i % (iter / 10) == 0) {
			j = min_square_error(h4, y);
			//println("=======================================");
			printf("iter: %8d  J: %12.8f", i, j);
			println();
		}
		
		Matrix& b0 = ms.backward(y);
		Matrix& b1 = s2.backward(b0);
		Matrix& b2 = f2.backward(b1);
		Matrix& b3 = s1.backward(b2);
		f1.backward(b3);
		
		f2.update(alpha);
		f1.update(alpha);
	}
	
	println();
	Matrix& h1 = f1.foward(x);
	Matrix& h2 = s1.foward(h1);
	Matrix& h3 = f2.foward(h2);
	Matrix& h4 = s2.foward(h3);
	
	println("Ans:");
	h4.print();
	println("ended");
}

void adult_data_gpu() {
	
	string data = ffull("files/adult.data");
	
	Grid g = Grid(data);
	//g.shuffle();
	
	GpuMatrix x = g.toMatrix(0, 14, true);
	CpuMatrix yy = g.toMatrix(14, 15, false);
	
	GpuMatrix y = yy.sltcols(0, 1);
	
	bool gpu = true;
	
	Affine f1 = Affine();
	f1.use_gpu(gpu);
	f1.set_dim(x.n, x.n);
	f1.init();
	
	Sigmoid s1 = Sigmoid();
	s1.use_gpu(gpu);
	s1.set_dim(f1.out_dim());
	s1.init();
	
	Affine f2 = Affine();
	f2.use_gpu(gpu);
	f2.set_dim(s1.out_dim(), y.n);
	f2.init();
	
	Sigmoid s2 = Sigmoid();
	s2.use_gpu(gpu);
	s2.set_dim(f2.out_dim());
	s2.init();
	
	Matrix& h1 = f1.foward(x);
	Matrix& h2 = s1.foward(h1);
	Matrix& h3 = f2.foward(h2);
	Matrix& h4 = s2.foward(h3);
	
	CpuMatrix h4c = gpu_cast(h4).cpu();
	CpuMatrix yc = y.cpu();
	hit(h4c, yc);
	
	println();
	
	println("About to train");
	int iter = 10;
	float j;
	float alpha = 0.001;
	for (int i = 0; i <= iter; i++) {
		//println("===================================");
		//f.print();
		
		Matrix& h1 = f1.foward(x);
		Matrix& h2 = s1.foward(h1);
		Matrix& h3 = f2.foward(h2);
		Matrix& h4 = s2.foward(h3);
		
		if (iter <= 10 || i % (iter / 10) == 0) {
			j = min_square_error(h4, y);
			//println("=======================================");
			printf("iter: %8d  J: %12.8f", i, j);
			println();
		}
		
		GpuMatrix dg = gpu_cast(h4) - y;
		Matrix& b1 = s2.backward(dg);
		Matrix& b2 = f2.backward(b1);
		Matrix& b3 = s1.backward(b2);
		f1.backward(b3);
		
		f2.update(alpha);
		f1.update(alpha);
		
		memtest();
	}
	
	println();
	Matrix& h21 = f1.foward(x);
	Matrix& h22 = s1.foward(h21);
	Matrix& h23 = f2.foward(h22);
	Matrix& h24 = s2.foward(h23);
	
	CpuMatrix h24c = gpu_cast(h24).cpu();
	CpuMatrix y2c = y.cpu();
	hit(h24c, y2c);
	
	memtest();
}

void adult_data_cpu() {
	
	string data = ffull("files/adult.data");
	
	Grid g = Grid(data);
	//g.shuffle();
	
	CpuMatrix x = g.toMatrix(0, 14, true);
	CpuMatrix yy = g.toMatrix(14, 15, false);
	
	CpuMatrix y = yy.sltcols(0, 1);
	
	bool gpu = false;
	
	Affine f1 = Affine();
	f1.use_gpu(gpu);
	f1.set_dim(x.n, x.n);
	f1.init();
	
	Sigmoid s1 = Sigmoid();
	s1.use_gpu(gpu);
	s1.set_dim(f1.out_dim());
	s1.init();
	
	Affine f2 = Affine();
	f2.use_gpu(gpu);
	f2.set_dim(s1.out_dim(), y.n);
	f2.init();
	
	Sigmoid s2 = Sigmoid();
	s2.use_gpu(gpu);
	s2.set_dim(f2.out_dim());
	s2.init();
	
	println("About to train");
	int iter = 10;
	float j;
	float alpha = 0.001;
	
	Matrix& h1 = f1.foward(x);
	Matrix& h2 = s1.foward(h1);
	Matrix& h3 = f2.foward(h2);
	Matrix& h4 = s2.foward(h3);
	
	hit(cpu_cast(h4), y);
	println();
	
	for (int i = 0; i <= iter; i++) {
		//println("===================================");
		//f.print();
		
		Matrix& h1 = f1.foward(x);
		Matrix& h2 = s1.foward(h1);
		Matrix& h3 = f2.foward(h2);
		Matrix& h4 = s2.foward(h3);
		
		if (iter <= 10 || i % (iter / 10) == 0) {
			j = min_square_error(h4, y);
			//println("=======================================");
			printf("iter: %8d  J: %12.8f", i, j);
			println();
		}
		
		CpuMatrix dg = cpu_cast(h4) - y;
		Matrix& b1 = s2.backward(dg);
		Matrix& b2 = f2.backward(b1);
		Matrix& b3 = s1.backward(b2);
		f1.backward(b3);
		
		f2.update(alpha);
		f1.update(alpha);
		
		memtest();
	}
	
	println();
	Matrix& h21 = f1.foward(x);
	Matrix& h22 = s1.foward(h21);
	Matrix& h23 = f2.foward(h22);
	Matrix& h24 = s2.foward(h23);
	
	hit(cpu_cast(h24), y);
	
	memtest();
	
}

void networktest() {
	string data = ffull("files/iris.data");
	
	Grid g = Grid(data);
	//g.shuffle();
	
	CpuMatrix x = g.toMatrix(0, 4, true);
	CpuMatrix y = g.toMatrix(4, 5, false);
	
	Network n = Network();
	n << Affine();
	n << Sigmoid();
	n << Affine();
	n << Sigmoid();
	n << MinSquare();
	
	n.init(x, y, false);
	
	println("CPU Training");
	println("================================================");
	float j = n.min_square_error();
	printf("iter: %8d  J: %12.8f", 0, j);
	println();
	for (int i = 0; i < 10; i++) {
		n.train(10000);
		j = n.min_square_error();
		printf("iter: %8d  J: %12.8f", i, j);
		println();
		
	}
	
	println("GPU Training");
	println("================================================");
	
	GpuMatrix gx = x;
	GpuMatrix gy = y;
	n.init(gx, gy, true);
	
	j = n.min_square_error();
	printf("iter: %8d  J: %12.8f", 0, j);
	fflush(stdout);
	for (int i = 0; i < 10; i++) {
		n.train(10000);
		j = n.min_square_error();
		printf("iter: %8d  J: %12.8f", i, j);
		fflush(stdout);
		println();
	}
	
	n.forward().print();
}

void net() {
	
	string data = ffull("files/iris.data");
	
	Grid g = Grid(data);
	//g.shuffle();
	
	CpuMatrix x = g.toMatrix(0, 4, false);
	
	CpuMatrix y = g.toMatrix(4, 5, false);
	
	Network n = Network();
	
	n << Affine();
	n << Sigmoid();
	n.init(x, y, false);
	
	n.forward();
	
	println();
	memtest();
}

int main(void) {
	
	println();
	
	net();
	memtest();
	println("ok");
	
	//adult_data_cpu();
	//networktest();
	//adult_data_gpu();
	//test_data();
	
	//iris_test_gpu();
	//sigmoid_test2();
	//gpu_test();
	//test2();
	
	return 0;
}


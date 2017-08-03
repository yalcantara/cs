/*
 * Network.h
 *
 *  Created on: Jun 24, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_NN_NETWORK_H_
#define CS_NN_NETWORK_H_


#include <cs/nn/Affine.h>
#include <cs/nn/Sigmoid.h>
#include <vector>

using namespace std;

namespace cs {
namespace nn {

class Network {
	
private:
	
	float alpha = 0.1;
	bool gpu = false;
	Matrix* x = nullptr;
	Matrix* y = nullptr;
	vector<Layer*> layers;
	
	const CpuMatrix cpu_last_grad()const;
	const GpuMatrix gpu_last_grad()const;
	
public:
	Network();
	virtual ~Network();
	
	void operator<<(Affine layer);
	void operator<<(Sigmoid layer);
	
	void set_alpha(float alpha);
	float get_alpha()const;
	void init(Matrix& x, Matrix& y, bool gpu);
	
	Matrix& forward();
	void backward();
	void update();
	
	void train(size_t iter);
	float min_square_error();
	
};

} // namespace nn
} // namespace cs 

#endif // NETWORK_H_

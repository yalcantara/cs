/*
 * gpu_layers.cuh
 *
 *  Created on: Feb 25, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_NN_GPU_LAYERS_CUH_
#define CS_NN_GPU_LAYERS_CUH_

#include <cs/math/GpuMatrix.h>

namespace cs {
using namespace math;
namespace nn {

void affine_dx(const GpuMatrix& x, const GpuMatrix& w, const GpuMatrix& dg, GpuMatrix& dx, GpuMatrix& dw, GpuVector& db);
void update_params(const GpuMatrix& w, const GpuMatrix& dw, float scalar);
void update_params(const GpuVector& b, const GpuVector& db, float scalar);


} // namespace nn
} // namespace cs

#endif // CS_NN_GPU_LAYERS_CUH_

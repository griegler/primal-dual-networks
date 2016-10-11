// Copyright (C) 2016 Gernot Riegler
// Institute for Computer Graphics and Vision (ICG)
// Graz University of Technology (TU GRAZ)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. All advertising materials mentioning features or use of this software
//    must display the following acknowledgement:
//    This product includes software developed by the ICG, TU GRAZ.
// 4. Neither the name of the ICG, TU GRAZ nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE PROVIDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



#include "utils.h"
#include "common.h"

#include "THC/THCTensorRandom.h"
#include "THCDeviceUtils.cuh"
#include <curand_kernel.h>

#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256
#define NUM_BLOCKS min((int)THCCeilDiv(size, (long) BLOCK_SIZE), MAX_NUM_BLOCKS)

__global__ void icgNoiseGaussian(float* out, const float* in, int size, 
    curandStateMtgp32* state, float mu, float sigma) {
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for(int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    float n = curand_normal(&state[blockIdx.x]);
    n = mu + sigma * n;
    if(i < size) {
      out[i] = in[i] + n;
    }
  }
}

__global__ void icgNoiseSpeckle(float* out, const float* in, int size, 
    curandStateMtgp32* state, float mu, float sigma) {
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for(int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    float n = curand_normal(&state[blockIdx.x]);
    n = mu + sigma * n;
    if(i < size) {
      float val = in[i];
      out[i] = val + val * n;
    }
  }
}

__global__ void icgNoiseLocalvar(float* out, const float* in, int size, 
    curandStateMtgp32* state, float sigma, float k, int inv) {
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for(int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    float n = curand_normal(&state[blockIdx.x]);
    if(i < size) {
      float val = in[i];
      float s = inv ? sigma / val : sigma * val;
      n = (s > 0 && val > 0) ? n * s : 0;
      out[i] = val + k * n;
    }
  }
}

__global__ void icgNoiseIntervalGaussian(float* out, const float* in, int size, 
    curandStateMtgp32* state, const float* interval, const float* mu, const float* sigma, int nelem) {
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for(int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    if(i < size) {
      float val = in[i];
      float n = 0;

      float from = -1e9;
      float to = interval[0];
      float ni = mu[1] + sigma[0] * curand_normal(&state[blockIdx.x]);
      n = val >= from && val < to ? ni : n;

      for(int idx = 0; idx < nelem - 1; ++idx) {
        from = interval[idx];
        to = interval[idx + 1];
        ni = mu[idx] + sigma[idx] * curand_normal(&state[blockIdx.x]);
        n = val >= from && val < to ? ni : n;
      }

      from = interval[nelem-1];
      to = 1e9;
      ni = mu[nelem] + sigma[nelem] * curand_normal(&state[blockIdx.x]);
      n = val >= from && val < to ? ni : n;
      
      out[i] = val + n;
    }
  }
}

static int icgcunn_IcgNoise_updateOutput(lua_State *L)
{
  THCState* state = getCutorchState(L);
  THCudaTensor* input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  int inplace = luaT_getfieldcheckboolean(L, 1, "inplace");

  const char* noise_type = luaT_getfieldcheckstring(L, 1, "noise_type");

  if(!inplace) {
    THCudaTensor_resizeAs(state, output, input);
  }

  float* in_data = THCudaTensor_data(state, input);
  float* out_data = inplace ? in_data : THCudaTensor_data(state, output);

  int n_dim = input->nDimension;
  long size = 1;
  for(int dim = 0; dim < n_dim; ++dim) {
    size = size * input->size[dim];
  }

  if(strcmp(noise_type, "gaussian") == 0 || strcmp(noise_type, "normal") == 0) {
    float mu = luaT_getfieldchecknumber(L, 1, "mu");
    float sigma = luaT_getfieldchecknumber(L, 1, "sigma");
    icgNoiseGaussian<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      out_data, in_data, size, 
      state->rngState->current_gen->gen_states, mu, sigma);
  }
  else if(strcmp(noise_type, "speckle") == 0) {
    float mu = luaT_getfieldchecknumber(L, 1, "mu");
    float sigma = luaT_getfieldchecknumber(L, 1, "sigma");
    icgNoiseSpeckle<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      out_data, in_data, size, 
      state->rngState->current_gen->gen_states, mu, sigma);
  }
  else if(strcmp(noise_type, "localvar") == 0) {
    float sigma = luaT_getfieldchecknumber(L, 1, "sigma");
    float k = luaT_getfieldchecknumber(L, 1, "k");
    int inverse = luaT_getfieldcheckboolean(L, 1, "inverse");
    icgNoiseLocalvar<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      out_data, in_data, size, 
      state->rngState->current_gen->gen_states, sigma, k, inverse);
  }
  else if(strcmp(noise_type, "intervalgaussian") == 0) {
    THCudaTensor* interval = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "interval", "torch.CudaTensor");
    THCudaTensor* mu = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "mu", "torch.CudaTensor");
    THCudaTensor* sigma = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "sigma", "torch.CudaTensor");
    long nelem = THCudaTensor_nElement(state, interval);
    luaL_argcheck(L, nelem > 0, 2, "#interval <= 0");
    luaL_argcheck(L, nelem+1 == THCudaTensor_nElement(state, mu), 2, "#interval+1 != #mu");
    luaL_argcheck(L, nelem+1 == THCudaTensor_nElement(state, sigma), 2, "#interval+1 != #sigma");

    float* interval_data = THCudaTensor_data(state, interval);
    float* mu_data = THCudaTensor_data(state, mu);
    float* sigma_data = THCudaTensor_data(state, sigma);
    icgNoiseIntervalGaussian<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      out_data, in_data, size, 
      state->rngState->current_gen->gen_states, interval_data, mu_data, sigma_data, nelem);
  }
  else {
    luaL_error(L, "unknown noise type %s", noise_type);
  }
  THCudaCheck(cudaGetLastError());

  if(inplace) {
    THCudaTensor_set(state, output, input);
  }

  return 1;
}


static const struct luaL_Reg icgcunn_IcgNoise__ [] = {
  {"IcgNoise_updateOutput", icgcunn_IcgNoise_updateOutput},
  {NULL, NULL}
};

void icgcunn_IcgNoise_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, icgcunn_IcgNoise__, "icgnn");
  lua_pop(L,1);
}

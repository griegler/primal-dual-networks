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


__global__ void icgThresholdForward(float* out, const float* in, 
    int length, float threshold, float val_greater, float val_smaller) {
  CUDA_KERNEL_LOOP(idx, length) {
    out[idx] = in[idx] > threshold ? val_greater : val_smaller;
  }
}


static int icgcunn_IcgThreshold_updateOutput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor* input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  float threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  float val_greater = luaT_getfieldchecknumber(L, 1, "val_greater");
  float val_smaller = luaT_getfieldchecknumber(L, 1, "val_smaller");
  THCudaTensor* output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  long nelem = THCudaTensor_nElement(state, input);

  THCudaTensor_resizeAs(state, output, input); 

  input = THCudaTensor_newContiguous(state, input);

  float* out = THCudaTensor_data(state, output);
  float* in = THCudaTensor_data(state, input);
  icgThresholdForward<<<GET_BLOCKS(nelem), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    out, in, nelem, threshold, val_greater, val_smaller);

  THCudaTensor_free(state, input);

  THCudaCheck(cudaGetLastError());
  return 1;
}


__global__ void icgThresholdBackward(float* grad_in, const int length) {
  CUDA_KERNEL_LOOP(idx, length) {
    grad_in[idx] = 0;
  }
}

static int icgcunn_IcgThreshold_updateGradInput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor* input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* grad_input = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

  long nelem = THCudaTensor_nElement(state, input);
  THCudaTensor_resizeAs(state, grad_input, input);

  float* grad_in = THCudaTensor_data(state, grad_input);
  icgThresholdBackward<<<GET_BLOCKS(nelem), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    grad_in, nelem);

  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg icgcunn_IcgThreshold__ [] = {
  {"IcgThreshold_updateOutput", icgcunn_IcgThreshold_updateOutput},
  {"IcgThreshold_updateGradInput", icgcunn_IcgThreshold_updateGradInput},
  {NULL, NULL}
};

void icgcunn_IcgThreshold_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, icgcunn_IcgThreshold__, "icgnn");
  lua_pop(L,1);
}

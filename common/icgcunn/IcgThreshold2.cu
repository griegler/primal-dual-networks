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


__global__ void icgThreshold2Forward(float* out, const float* in1, 
    const float* in2, int length, float threshold, float val) {
  CUDA_KERNEL_LOOP(idx, length) {
    out[idx] = in2[idx] <= threshold ? val : in1[idx];
  }
}


static int icgcunn_IcgThreshold2_updateOutput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor* input1 = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* input2 = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  float threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  float val = luaT_getfieldchecknumber(L, 1, "val");
  THCudaTensor* output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  long nelem = THCudaTensor_nElement(state, input1);
  luaL_argcheck(L, nelem == THCudaTensor_nElement(state, input2), 2, "input1 should have same number of elements as input2");

  THCudaTensor_resizeAs(state, output, input1); 

  input1 = THCudaTensor_newContiguous(state, input1);
  input2 = THCudaTensor_newContiguous(state, input2);

  float* out = THCudaTensor_data(state, output);
  float* in1 = THCudaTensor_data(state, input1);
  float* in2 = THCudaTensor_data(state, input2);
  icgThreshold2Forward<<<GET_BLOCKS(nelem), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    out, in1, in2, nelem, threshold, val);

  THCudaTensor_free(state, input1);
  THCudaTensor_free(state, input2);

  THCudaCheck(cudaGetLastError());
  return 1;
}


__global__ void icgThreshold2Backward(float* grad_in1, const float* grad_out, 
    const float* in2, int length, float threshold) {
  CUDA_KERNEL_LOOP(idx, length) {
    grad_in1[idx] = in2[idx] <= threshold ? 0 : grad_out[idx];
  }
}

static int icgcunn_IcgThreshold2_updateGradInput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor* input1 = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* input2 = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor* grad_input1 = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor* grad_output = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
  float threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  float val = luaT_getfieldchecknumber(L, 1, "val");

  THAssert(THCudaTensor_checkGPU(state, 4, input2, grad_output, grad_input1, input1));
  
  long nelem = THCudaTensor_nElement(state, input1);

  THCudaTensor_resizeAs(state, grad_input1, input1);

  float* grad_out = THCudaTensor_data(state, grad_output);
  float* grad_in1 = THCudaTensor_data(state, grad_input1);
  float* in2 = THCudaTensor_data(state, input2);
  icgThreshold2Backward<<<GET_BLOCKS(nelem), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    grad_in1, grad_out, in2, nelem, threshold);

  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg icgcunn_IcgThreshold2__ [] = {
  {"IcgThreshold2_updateOutput", icgcunn_IcgThreshold2_updateOutput},
  {"IcgThreshold2_updateGradInput", icgcunn_IcgThreshold2_updateGradInput},
  {NULL, NULL}
};

void icgcunn_IcgThreshold2_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, icgcunn_IcgThreshold2__, "icgnn");
  lua_pop(L,1);
}

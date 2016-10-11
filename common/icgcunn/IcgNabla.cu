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


__global__ void icgNablaForward(float* out, const float* in, int length, int height, 
    int width) {

  int offset = height * width;
  CUDA_KERNEL_LOOP(out_idx, length) {
    int h = out_idx / width;
    int w = out_idx % width;

    out[out_idx] = w < width - 1 ? in[out_idx + 1] - in[out_idx] : 0;
    out[out_idx + offset] = h < height - 1 ? in[out_idx + width] - in[out_idx] : 0;
  }
}


static int icgcunn_IcgNabla_updateOutput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor* in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  long n_dim = in->nDimension;
  luaL_argcheck(L, n_dim == 3 || n_dim == 4, 2, "3D or 4D(batch mode) tensor expected");

  long num, channels, height, width;
  if(n_dim == 3) {
    num = 1;
    channels = THCudaTensor_size(state, in, 0);
    height = THCudaTensor_size(state, in, 1);
    width = THCudaTensor_size(state, in, 2);
    THCudaTensor_resize3d(state, out, 2 * channels, height, width);
  }
  else if(n_dim == 4) {
    num = THCudaTensor_size(state, in, 0);
    channels = THCudaTensor_size(state, in, 1);
    height = THCudaTensor_size(state, in, 2);
    width = THCudaTensor_size(state, in, 3);
    THCudaTensor_resize4d(state, out, num, 2 * channels, height, width);
  }

  long length = height * width;
  in = THCudaTensor_newContiguous(state, in);

  float* out_data = THCudaTensor_data(state, out);
  float* in_data = THCudaTensor_data(state, in);
  for(long n = 0; n < num; ++n) {
    for(long c = 0; c < channels; ++c) {
      icgNablaForward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        out_data, in_data, length, height, width);
      out_data = out_data + 2 * length;
      in_data = in_data + length;
    }
  }

  THCudaTensor_free(state, in);

  THCudaCheck(cudaGetLastError());
  return 1;
}


__global__ void icgNablaBackward(float* grad_in, const float* grad_out,
    int length, int height, int width) {

  int offset = height * width;
  CUDA_KERNEL_LOOP(idx, length) {
    int h = idx / width;
    int w = idx % width;

    float grad = 0;
    grad = grad + (w > 0 ? grad_out[idx - 1] : 0);
    grad = grad + (w < width - 1 ? -grad_out[idx] : 0);

    grad = grad + (h > 0 ? grad_out[idx + offset - width] : 0);
    grad = grad + (h < height - 1 ? -grad_out[idx + offset] : 0);

    grad_in[idx] = grad;
  }
}

static int icgcunn_IcgNabla_updateGradInput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor* in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* grad_out = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor* out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor* grad_in = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, in, out, grad_in, grad_out));
  
  long n_dim = in->nDimension;
  
  long num, channels, height, width;
  if(n_dim == 3) {
    num = 1;
    channels = THCudaTensor_size(state, in, 0);
    height = THCudaTensor_size(state, in, 1);
    width = THCudaTensor_size(state, in, 2);
  }
  else if(n_dim == 4) {
    num = THCudaTensor_size(state, in, 0);
    channels = THCudaTensor_size(state, in, 1);
    height = THCudaTensor_size(state, in, 2);
    width = THCudaTensor_size(state, in, 3);
  }

  long length = height * width;
  THCudaTensor_resizeAs(state, grad_in, in);

  float* grad_in_data = THCudaTensor_data(state, grad_in);
  float* grad_out_data = THCudaTensor_data(state, grad_out);
  for(long n = 0; n < num; ++n) {
    for(long c = 0; c < channels; ++c) {
      icgNablaBackward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        grad_in_data, grad_out_data, length, height, width);
      grad_in_data = grad_in_data + length;
      grad_out_data = grad_out_data + 2 * length;
    }
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg icgcunn_IcgNabla__ [] = {
  {"IcgNabla_updateOutput", icgcunn_IcgNabla_updateOutput},
  {"IcgNabla_updateGradInput", icgcunn_IcgNabla_updateGradInput},
  {NULL, NULL}
};

void icgcunn_IcgNabla_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, icgcunn_IcgNabla__, "icgnn");
  lua_pop(L,1);
}

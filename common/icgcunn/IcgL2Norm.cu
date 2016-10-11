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


__global__ void icgL2NormForward(float* out, const float* in, int length, 
    int in_channels, int out_channels, int height, int width) {

  int offset = height * width;
  CUDA_KERNEL_LOOP(out_idx, length) {
    float norm = 0;
    for(int c = 0; c < in_channels; ++c) {
      float in_val = in[out_idx + c * offset];
      norm = norm + in_val * in_val;
    }
    float out_val = sqrt(norm);

    for(int c = 0; c < out_channels; ++c) {
      out[out_idx + c * offset] = out_val;
    }
  }
}

static int icgcunn_IcgL2Norm_updateOutput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor *in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  int spatial = luaT_getfieldcheckboolean(L, 1, "spatial");
  THCudaTensor *out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  long n_dim = in->nDimension;
  luaL_argcheck(L, n_dim == 3 || n_dim == 4, 2, "3D or 4D(batch mode) tensor expected");

  long num, channels, height, width, out_channels;
  if(n_dim == 3) {
    num = 1;
    channels = THCudaTensor_size(state, in, 0);
    height = THCudaTensor_size(state, in, 1);
    width = THCudaTensor_size(state, in, 2);
    out_channels = spatial ? channels : 1;
    THCudaTensor_resize3d(state, out, out_channels, height, width);
  }
  else if(n_dim == 4) {
    num = THCudaTensor_size(state, in, 0);
    channels = THCudaTensor_size(state, in, 1);
    height = THCudaTensor_size(state, in, 2);
    width = THCudaTensor_size(state, in, 3);
    out_channels = spatial ? channels : 1;
    THCudaTensor_resize4d(state, out, num, out_channels, height, width);
  }

  long length = height * width;
  in = THCudaTensor_newContiguous(state, in);

  float* in_data = THCudaTensor_data(state, in);
  float* out_data = THCudaTensor_data(state, out);
  for(long n = 0; n < num; ++n) {
    icgL2NormForward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      out_data, in_data, length, channels, out_channels, height, width);
    out_data = out_data + out_channels * length;
    in_data = in_data + channels * length;
  }

  THCudaTensor_free(state, in);

  THCudaCheck(cudaGetLastError());
  return 1;
}


__global__ void icgL2NormBackward(float* grad_in, const float* in, 
    const float* out, const float* grad_out,
    int length, int in_channels, int out_channels, int height, int width, float dft) {

  CUDA_KERNEL_LOOP(in_idx, length) {
    int hw = in_idx % (height * width);
    int h = hw / width;
    int w = hw % width;

    float in_val = in[in_idx];

    float grad_in_val = 0;
    for(int co = 0; co < out_channels; ++co) {
      int out_idx = (co * height + h) * width + w;
      float out_val = out[out_idx];
      out_val = out_val == 0 ? dft : out_val;
      grad_in_val = grad_in_val + grad_out[out_idx] * in_val / out_val;
    }

    grad_in[in_idx] = grad_in_val;
  }
}


static int icgcunn_IcgL2Norm_updateGradInput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor* in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* grad_out = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor* out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  int spatial = luaT_getfieldcheckboolean(L, 1, "spatial");
  float dft = luaT_getfieldchecknumber(L, 1, "dft");
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
  long out_channels = spatial ? channels : 1;
  THCudaTensor_resizeAs(state, grad_in, in);

  float* grad_in_data = THCudaTensor_data(state, grad_in);
  float* in_data = THCudaTensor_data(state, in);
  float* out_data = THCudaTensor_data(state, out);
  float* grad_out_data = THCudaTensor_data(state, grad_out);
  long length = channels * height * width;
  for(long n = 0; n < num; ++n) {
    icgL2NormBackward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      grad_in_data, in_data, out_data, grad_out_data, length, channels, out_channels, height, width, dft);

    grad_in_data = grad_in_data + channels * height * width;
    in_data = in_data + channels * height * width;
    grad_out_data = grad_out_data + out_channels * height * width;
    out_data = out_data + out_channels * height * width;
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg icgcunn_IcgL2Norm__ [] = {
  {"IcgL2Norm_updateOutput", icgcunn_IcgL2Norm_updateOutput},
  {"IcgL2Norm_updateGradInput", icgcunn_IcgL2Norm_updateGradInput},
  {NULL, NULL}
};

void icgcunn_IcgL2Norm_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, icgcunn_IcgL2Norm__, "icgnn");
  lua_pop(L,1);
}

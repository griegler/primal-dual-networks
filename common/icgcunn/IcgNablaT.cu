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


__global__ void icgNablaTForward(float* out, const float* in, int length, int height, 
    int width, int neg) {

  int offset = height * width;
  CUDA_KERNEL_LOOP(idx, length) {
    int h = idx / width;
    int w = idx % width;

    float val = 0;
    val = val + (w < width - 1 ? -in[idx] : 0);
    val = val + (w > 0 ? in[idx - 1] : 0);
    val = val + (h < height - 1 ? -in[idx + offset] : 0);
    val = val + (h > 0 ? in[idx + offset - width] : 0);

    out[idx] = neg ? -val : val;
  }
}

static int icgcunn_IcgNablaT_updateOutput(lua_State *L)
{
  THCState* state = getCutorchState(L);
  THCudaTensor *in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  int neg = luaT_getfieldcheckboolean(L, 1, "neg");
  THCudaTensor *out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  long n_dim = in->nDimension;
  luaL_argcheck(L, n_dim == 3 || n_dim == 4, 2, "3D or 4D(batch mode) tensor expected");

  long num, channels, height, width, out_channels;
  if(n_dim == 3) {
    num = 1;
    channels = THCudaTensor_size(state, in, 0);
    height = THCudaTensor_size(state, in, 1);
    width = THCudaTensor_size(state, in, 2);
    out_channels = channels / 2;
    THCudaTensor_resize3d(state, out, out_channels, height, width);
  }
  else if(n_dim == 4) {
    num = THCudaTensor_size(state, in, 0);
    channels = THCudaTensor_size(state, in, 1);
    height = THCudaTensor_size(state, in, 2);
    width = THCudaTensor_size(state, in, 3);
    out_channels = channels / 2;
    THCudaTensor_resize4d(state, out, num, out_channels, height, width);
  }
  luaL_argcheck(L, channels % 2 == 0, 1, "channels % 2 != 0");

  long length = height * width;
  in = THCudaTensor_newContiguous(state, in);

  float* out_data = THCudaTensor_data(state, out);
  float* in_data = THCudaTensor_data(state, in);
  for(long n = 0; n < num; ++n) {
    for(long c = 0; c < out_channels; ++c) {
      icgNablaTForward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        out_data, in_data, length, height, width, neg);
      out_data = out_data + length;
      in_data = in_data + 2 * length;
    }
  }
  THCudaTensor_free(state, in);

  THCudaCheck(cudaGetLastError());
  return 1;
}


__global__ void icgNablaTBackward(float* grad_in, const float* grad_out,
    int n, int height, int width, int neg) {

  int offset = height * width;
  CUDA_KERNEL_LOOP(out_idx, n) {
    int h = out_idx / width;
    int w = out_idx % width;

    float grad_x = (w >= 0 && w < width - 1) ? 
        grad_out[out_idx + 1] - grad_out[out_idx] : 0;
    float grad_y = (h >= 0 && h < height - 1) ? 
        grad_out[out_idx + width] - grad_out[out_idx] : 0;

    grad_in[out_idx] = neg ? -grad_x : grad_x;
    grad_in[out_idx + offset] = neg ? -grad_y : grad_y;
  }
}


static int icgcunn_IcgNablaT_updateGradInput(lua_State *L)
{
  THCState* state = getCutorchState(L);
  THCudaTensor* in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* grad_out = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int neg = luaT_getfieldcheckboolean(L, 1, "neg");
  THCudaTensor* out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor* grad_in = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, in, out, grad_in, grad_out));
  
  long n_dim = in->nDimension;

  long num, channels, height, width, out_channels;
  if(n_dim == 3) {
    num = 1;
    channels = THCudaTensor_size(state, in, 0);
    height = THCudaTensor_size(state, in, 1);
    width = THCudaTensor_size(state, in, 2);
    out_channels = channels / 2;
  }
  else if(n_dim == 4) {
    num = THCudaTensor_size(state, in, 0);
    channels = THCudaTensor_size(state, in, 1);
    height = THCudaTensor_size(state, in, 2);
    width = THCudaTensor_size(state, in, 3);
    out_channels = channels / 2;
  }
  THCudaTensor_resizeAs(state, grad_in, in);

  long length = height * width;
  
  float* grad_in_data = THCudaTensor_data(state, grad_in);
  float* grad_out_data = THCudaTensor_data(state, grad_out);
  for(long n = 0; n < num; ++n) {
    for(long c = 0; c < out_channels; ++c) {
      icgNablaTBackward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        grad_in_data, grad_out_data, length, height, width, neg);
      grad_in_data = grad_in_data + 2 * length;
      grad_out_data = grad_out_data + length;
    }
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg icgcunn_IcgNablaT__ [] = {
  {"IcgNablaT_updateOutput", icgcunn_IcgNablaT_updateOutput},
  {"IcgNablaT_updateGradInput", icgcunn_IcgNablaT_updateGradInput},
  {NULL, NULL}
};

void icgcunn_IcgNablaT_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, icgcunn_IcgNablaT__, "icgnn");
  lua_pop(L,1);
}

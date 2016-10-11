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


__global__ void icgGeneralizedNablaTForward(float* out, const float* in, int length, int height, 
    int width, int neg, float* dir_data, int n_dir) {

  // int offset = height * width;
  CUDA_KERNEL_LOOP(idx, length) {
    int h = idx / width;
    int w = idx % width;

    float val = 0;
    for(int dir_idx = 0; dir_idx < n_dir; ++dir_idx) {
      int dir_x = -dir_data[dir_idx];
      int dir_y = -dir_data[n_dir + dir_idx];

      int in_idx = (dir_idx * height + h) * width + w;

      float add1 = (w+dir_x >= 0 && h+dir_y >= 0 && w+dir_x < width && h+dir_y < height) ? in[in_idx + dir_y * width + dir_x] : 0;
      float add2 = (w-dir_x >= 0 && h-dir_y >= 0 && w-dir_x < width && h-dir_y < height) ? -in[in_idx] : 0;
      val = val + add1 + add2;
    }

    out[idx] = neg ? -val : val;
  }
}

static int icgcunn_IcgGeneralizedNablaT_updateOutput(lua_State *L)
{
  THCState* state = getCutorchState(L);
  THCudaTensor *in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  int neg = luaT_getfieldcheckboolean(L, 1, "neg");
  THCudaTensor* dir = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "directions", "torch.CudaTensor");
  THCudaTensor *out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  long n_dim = in->nDimension;
  luaL_argcheck(L, n_dim == 3 || n_dim == 4, 2, "3D or 4D(batch mode) tensor expected");

  long n_dir = THCudaTensor_size(state, dir, 1);

  long num, channels, height, width, out_channels;
  if(n_dim == 3) {
    num = 1;
    channels = THCudaTensor_size(state, in, 0);
    height = THCudaTensor_size(state, in, 1);
    width = THCudaTensor_size(state, in, 2);
    out_channels = channels / n_dir;
    THCudaTensor_resize3d(state, out, out_channels, height, width);
  }
  else if(n_dim == 4) {
    num = THCudaTensor_size(state, in, 0);
    channels = THCudaTensor_size(state, in, 1);
    height = THCudaTensor_size(state, in, 2);
    width = THCudaTensor_size(state, in, 3);
    out_channels = channels / n_dir;
    THCudaTensor_resize4d(state, out, num, out_channels, height, width);
  }
  luaL_argcheck(L, channels % n_dir == 0, 1, "channels % n_dir != 0");

  long length = height * width;
  in = THCudaTensor_newContiguous(state, in);

  float* out_data = THCudaTensor_data(state, out);
  float* in_data = THCudaTensor_data(state, in);
  float* dir_data = THCudaTensor_data(state, dir);
  for(long n = 0; n < num; ++n) {
    for(long c = 0; c < out_channels; ++c) {
      icgGeneralizedNablaTForward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        out_data, in_data, length, height, width, neg, dir_data, n_dir);
      out_data = out_data + length;
      in_data = in_data + n_dir * length;
    }
  }
  THCudaTensor_free(state, in);

  THCudaCheck(cudaGetLastError());
  return 1;
}


__global__ void icgGeneralizedNablaTBackward(float* grad_in, const float* grad_out,
    int n, int height, int width, int neg, float* dir_data, int n_dir) {

  // int offset = height * width;
  CUDA_KERNEL_LOOP(out_idx, n) {
    int h = out_idx / width;
    int w = out_idx % width;

    for(int dir_idx = 0; dir_idx < n_dir; ++dir_idx) {
      int dir_x = -dir_data[dir_idx];
      int dir_y = -dir_data[n_dir + dir_idx];
      
      int in_idx = ((dir_idx) * height + h) * width + w;

      if(w+dir_x >= 0 && h+dir_y >= 0 && w+dir_x < width && h+dir_y < height) {
        // grad_in[in_idx + dir_y * width + dir_x] += neg * grad_out[out_idx];
        atomicAdd(grad_in + in_idx + dir_y * width + dir_x, neg * grad_out[out_idx]);
      }
      if(w-dir_x >= 0 && h-dir_y >= 0 && w-dir_x < width && h-dir_y < height) {
        // grad_in[in_idx] += -neg * grad_out[out_idx];
        atomicAdd(grad_in + in_idx, -neg * grad_out[out_idx]);
      }
    }
  }
}


static int icgcunn_IcgGeneralizedNablaT_updateGradInput(lua_State *L)
{
  THCState* state = getCutorchState(L);
  THCudaTensor* in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* grad_out = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int neg = luaT_getfieldcheckboolean(L, 1, "neg");
  THCudaTensor* dir = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "directions", "torch.CudaTensor");
  THCudaTensor* out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor* grad_in = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, in, out, grad_in, grad_out));
  
  long n_dim = in->nDimension;
  long n_dir = THCudaTensor_size(state, dir, 1);

  long num, channels, height, width, out_channels;
  if(n_dim == 3) {
    num = 1;
    channels = THCudaTensor_size(state, in, 0);
    height = THCudaTensor_size(state, in, 1);
    width = THCudaTensor_size(state, in, 2);
    out_channels = channels / n_dir;
  }
  else if(n_dim == 4) {
    num = THCudaTensor_size(state, in, 0);
    channels = THCudaTensor_size(state, in, 1);
    height = THCudaTensor_size(state, in, 2);
    width = THCudaTensor_size(state, in, 3);
    out_channels = channels / n_dir;
  }
  THCudaTensor_resizeAs(state, grad_in, in);

  long length = height * width;
  
  float* grad_in_data = THCudaTensor_data(state, grad_in);
  float* grad_out_data = THCudaTensor_data(state, grad_out);
  float* dir_data = THCudaTensor_data(state, dir);

  THCudaTensor_zero(state, grad_in);
  neg = neg ? -1 : 1;

  for(long n = 0; n < num; ++n) {
    for(long c = 0; c < out_channels; ++c) {
      icgGeneralizedNablaTBackward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        grad_in_data, grad_out_data, length, height, width, neg, dir_data, n_dir);
      grad_in_data = grad_in_data + n_dir * length;
      grad_out_data = grad_out_data + length;
    }
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg icgcunn_IcgGeneralizedNablaT__ [] = {
  {"IcgGeneralizedNablaT_updateOutput", icgcunn_IcgGeneralizedNablaT_updateOutput},
  {"IcgGeneralizedNablaT_updateGradInput", icgcunn_IcgGeneralizedNablaT_updateGradInput},
  {NULL, NULL}
};

void icgcunn_IcgGeneralizedNablaT_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, icgcunn_IcgGeneralizedNablaT__, "icgnn");
  lua_pop(L,1);
}

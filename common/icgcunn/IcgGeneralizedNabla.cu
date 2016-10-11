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


__global__ void icgGeneralizedNablaForward(float* out, const float* in, const float* dir, 
    int length, int height, int width, int n_dir) {

  int offset = height * width;
  CUDA_KERNEL_LOOP(idx, length) {
    int h = idx / width;
    int w = idx % width;

    for(int dir_idx = 0; dir_idx < n_dir; ++dir_idx) {
      int dir_x = dir[dir_idx];
      int dir_y = dir[n_dir + dir_idx];

      int out_idx = idx + dir_idx * offset;

      if(w >= -dir_x && w < width - dir_x && h >= -dir_y && h < height - dir_y) {
        out[out_idx] = in[idx + dir_y * width + dir_x] - in[idx];
      }
      else {
        out[out_idx] = 0;
      }
    }
  }
}


static int icgcunn_IcgGeneralizedNabla_updateOutput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor* in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* dir = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "directions", "torch.CudaTensor");
  THCudaTensor* out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  long n_dim = in->nDimension;
  luaL_argcheck(L, n_dim == 3 || n_dim == 4, 2, "3D or 4D(batch mode) tensor expected");

  long n_dir = THCudaTensor_size(state, dir, 1);

  long num, channels, height, width;
  if(n_dim == 3) {
    num = 1;
    channels = THCudaTensor_size(state, in, 0);
    height = THCudaTensor_size(state, in, 1);
    width = THCudaTensor_size(state, in, 2);
    THCudaTensor_resize3d(state, out, n_dir * channels, height, width);
  }
  else if(n_dim == 4) {
    num = THCudaTensor_size(state, in, 0);
    channels = THCudaTensor_size(state, in, 1);
    height = THCudaTensor_size(state, in, 2);
    width = THCudaTensor_size(state, in, 3);
    THCudaTensor_resize4d(state, out, num, n_dir * channels, height, width);
  }

  long length = height * width;
  in = THCudaTensor_newContiguous(state, in);

  float* out_data = THCudaTensor_data(state, out);
  float* in_data = THCudaTensor_data(state, in);
  float* dir_data = THCudaTensor_data(state, dir);
  for(long n = 0; n < num; ++n) {
    for(long c = 0; c < channels; ++c) {
      icgGeneralizedNablaForward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        out_data, in_data, dir_data, length, height, width, n_dir);
      out_data = out_data + n_dir * length;
      in_data = in_data + length;
    }
  }

  THCudaTensor_free(state, in);

  THCudaCheck(cudaGetLastError());
  return 1;
}


__global__ void icgGeneralizedNablaBackward(float* grad_in, const float* grad_out, const float* dir,
    int length, int height, int width, int n_dir) {

  int offset = height * width;
  CUDA_KERNEL_LOOP(idx, length) {
    int h = idx / width;
    int w = idx % width;

    float grad = 0;
    for(int dir_idx = 0; dir_idx < n_dir; ++dir_idx) {
      long dir_x = dir[dir_idx];
      long dir_y = dir[n_dir + dir_idx];

      long out_idx = idx + dir_idx * offset; 

      grad = grad + ((w >= dir_x && w < width+dir_x && h >= dir_y && h < height+dir_y) ? grad_out[out_idx - dir_y * width - dir_x] : 0);
      grad = grad + ((w >= -dir_x && w < width-dir_x && h >= -dir_y && h < height-dir_y) ? -grad_out[out_idx] : 0);
    }
    grad_in[idx] = grad;
  }
}

static int icgcunn_IcgGeneralizedNabla_updateGradInput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor* in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* grad_out = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor* out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor* dir = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "directions", "torch.CudaTensor");
  THCudaTensor* grad_in = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, in, out, grad_in, grad_out));

  long n_dir = THCudaTensor_size(state, dir, 1);
  
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
  float* dir_data = THCudaTensor_data(state, dir);
  for(long n = 0; n < num; ++n) {
    for(long c = 0; c < channels; ++c) {
      icgGeneralizedNablaBackward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        grad_in_data, grad_out_data, dir_data, length, height, width, n_dir);
      grad_in_data = grad_in_data + length;
      grad_out_data = grad_out_data + n_dir * length;
    }
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg icgcunn_IcgGeneralizedNabla__ [] = {
  {"IcgGeneralizedNabla_updateOutput", icgcunn_IcgGeneralizedNabla_updateOutput},
  {"IcgGeneralizedNabla_updateGradInput", icgcunn_IcgGeneralizedNabla_updateGradInput},
  {NULL, NULL}
};

void icgcunn_IcgGeneralizedNabla_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, icgcunn_IcgGeneralizedNabla__, "icgnn");
  lua_pop(L,1);
}

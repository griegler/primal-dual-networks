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


__global__ void icgMaskGridForward(float* out, const float* in, float mask_value, int length, int height, 
    int width, int height_factor, int width_factor, int h_offset, int w_offset) {

  CUDA_KERNEL_LOOP(out_idx, length) {
    int h = out_idx / width;
    int w = out_idx % width;

    bool opt = (h % height_factor == h_offset) && (w % width_factor == w_offset);
    out[out_idx] = opt ? in[out_idx] : mask_value;
  }
}

__global__ void icgMaskBorderForward(float* out, const float* in, float mask_value, int length, int height, 
    int width, int border) {

  CUDA_KERNEL_LOOP(out_idx, length) {
    int h = out_idx / width;
    int w = out_idx % width;

    bool opt = (h >= border && h < height - border && w >= border && w < width - border);
    out[out_idx] = opt ? in[out_idx] : mask_value;
  }
}


static int icgcunn_IcgMask_updateOutput(lua_State *L)
{
  THCState* state = getCutorchState(L);
  THCudaTensor* in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  const char* mask_type = luaT_getfieldcheckstring(L, 1, "mask_type");
  float mask_value = luaT_getfieldchecknumber(L, 1, "mask_value");

  
  int height_factor, width_factor;
  int border;
  if(strcmp(mask_type, "grid") == 0) {
    height_factor = luaT_getfieldcheckint(L, 1, "height_factor");
    width_factor = luaT_getfieldcheckint(L, 1, "width_factor");
  }
  else if(strcmp(mask_type, "border") == 0) {
    border = luaT_getfieldcheckint(L, 1, "border");
  }
  else {
    luaL_error(L, "unknown mask type: %s", mask_type);
  }


  long n_dim = in->nDimension;
  luaL_argcheck(L, n_dim == 3 || n_dim == 4, 2, "3D or 4D(batch mode) tensor expected");

  long num, channels, height, width;
  if(n_dim == 3) {
    num = 1;
    channels = THCudaTensor_size(state, in, 0);
    height = THCudaTensor_size(state, in, 1);
    width = THCudaTensor_size(state, in, 2);
    THCudaTensor_resize3d(state, out, channels, height, width);
  }
  else if(n_dim == 4) {
    num = THCudaTensor_size(state, in, 0);
    channels = THCudaTensor_size(state, in, 1);
    height = THCudaTensor_size(state, in, 2);
    width = THCudaTensor_size(state, in, 3);
    THCudaTensor_resize4d(state, out, num, channels, height, width);
  }

  long length = height * width;
  in = THCudaTensor_newContiguous(state, in);

  if(strcmp(mask_type, "grid") == 0) {
    long h_offset = (height_factor - 1) / 2;
    long w_offset = (width_factor - 1) / 2;

    float* out_data = THCudaTensor_data(state, out);
    float* in_data = THCudaTensor_data(state, in);
    for(long n = 0; n < num; ++n) {
      for(long c = 0; c < channels; ++c) {
        icgMaskGridForward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          out_data, in_data, mask_value, length, height, width, height_factor, width_factor, h_offset, w_offset);
        out_data = out_data + length;
        in_data = in_data + length;
      }
    }
  }
  else if(strcmp(mask_type, "border") == 0) {
    float* out_data = THCudaTensor_data(state, out);
    float* in_data = THCudaTensor_data(state, in);
    for(long n = 0; n < num; ++n) {
      for(long c = 0; c < channels; ++c) {
        icgMaskBorderForward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          out_data, in_data, mask_value, length, height, width, border);
        out_data = out_data + length;
        in_data = in_data + length;
      }
    }
  }

  THCudaTensor_free(state, in);

  THCudaCheck(cudaGetLastError());
  return 1;
}


static int icgcunn_IcgMask_updateGradInput(lua_State *L)
{
  THCState* state = getCutorchState(L);
  THCudaTensor* in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* grad_out = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor* out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor* grad_in = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, in, out, grad_in, grad_out));

  const char* mask_type = luaT_getfieldcheckstring(L, 1, "mask_type");
  
  int height_factor, width_factor;
  int border;
  if(strcmp(mask_type, "grid") == 0) {
    height_factor = luaT_getfieldcheckint(L, 1, "height_factor");
    width_factor = luaT_getfieldcheckint(L, 1, "width_factor");
  }
  else if(strcmp(mask_type, "border") == 0) {
    border = luaT_getfieldcheckint(L, 1, "border");
  }
  else {
    luaL_error(L, "unknown mask type: %s", mask_type);
  }
  
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

  THCudaTensor_resizeAs(state, grad_in, in);

  long length = height * width;
  
  if(strcmp(mask_type, "grid") == 0) {
    long h_offset = (height_factor - 1) / 2;
    long w_offset = (width_factor - 1) / 2;

    float* grad_in_data = THCudaTensor_data(state, grad_in);
    float* grad_out_data = THCudaTensor_data(state, grad_out);
    for(long n = 0; n < num; ++n) {
      for(long c = 0; c < channels; ++c) {
        icgMaskGridForward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          grad_in_data, grad_out_data, 0, length, height, width, height_factor, width_factor, h_offset, w_offset);
        grad_in_data = grad_in_data + length;
        grad_out_data = grad_out_data + length;
      }
    }
  }
  else if(strcmp(mask_type, "border") == 0) {
    float* grad_in_data = THCudaTensor_data(state, grad_in);
    float* grad_out_data = THCudaTensor_data(state, grad_out);
    for(long n = 0; n < num; ++n) {
      for(long c = 0; c < channels; ++c) {
        icgMaskBorderForward<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          grad_in_data, grad_out_data, 0, length, height, width, border);
        grad_in_data = grad_in_data + length;
        grad_out_data = grad_out_data + length;
      }
    }
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg icgcunn_IcgMask__ [] = {
  {"IcgMask_updateOutput", icgcunn_IcgMask_updateOutput},
  {"IcgMask_updateGradInput", icgcunn_IcgMask_updateGradInput},
  {NULL, NULL}
};

void icgcunn_IcgMask_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, icgcunn_IcgMask__, "icgnn");
  lua_pop(L,1);
}

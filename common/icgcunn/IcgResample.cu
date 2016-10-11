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


__device__ int clampCoord(int w, int width) {
  w = w < 0 ? 0 : w;
  w = w >= width ? width - 1 : w;
  return w;
}

__device__ float getP(const float* in_data, int h, int w, int height, int width) {
  h = clampCoord(h, height);
  w = clampCoord(w, width);

  int in_idx = (h * width) + w;
  return in_data[in_idx];
}


template <int inter_kernel>
__global__ void icgResampleForward(float* out, const float* in, int out_length, 
    int in_height, int in_width, int out_height, int out_width, float height_factor, float width_factor,
    int kernel_width, float kernel_scale) {

  int kernel_width_half = ceil(kernel_width / 2.0);

  CUDA_KERNEL_LOOP(out_idx, out_length) {
    float yo = (out_idx / out_width) + 0.5;
    float xo = (out_idx % out_width) + 0.5;

    float yi = yo / height_factor;
    float xi = xo / width_factor;

    float val = 0;
    int kh;
    for(kh = -kernel_width_half+1; kh <= kernel_width_half; ++kh) {
      int kw;
      for(kw = -kernel_width_half+1; kw <= kernel_width_half; ++kw) {
        float y = round(yi) + kh - 0.5;
        float x = round(xi) + kw - 0.5;
        
        float xdiff = kernel_scale * (xi - x);
        float ydiff = kernel_scale * (yi - y);

        float kx_val, ky_val;
        if(inter_kernel == 0) {
          kx_val = (-0.5 <= xdiff) && (xdiff < 0.5);
          ky_val = (-0.5 <= ydiff) && (ydiff < 0.5);
        }
        if(inter_kernel == 1) {
          kx_val = (xdiff+1) * ((-1 <= xdiff) && (xdiff < 0)) + (1-xdiff) * ((0 <= xdiff) & (xdiff <= 1));
          ky_val = (ydiff+1) * ((-1 <= ydiff) && (ydiff < 0)) + (1-ydiff) * ((0 <= ydiff) & (ydiff <= 1));
        }
        if(inter_kernel == 2) {
          float absx = fabs(xdiff);
          float absx2 = absx*absx;
          float absx3 = absx2*absx;
          float absy = fabs(ydiff);
          float absy2 = absy*absy;
          float absy3 = absy2*absy;
          kx_val = (1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) + (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) && (absx <= 2));
          ky_val = (1.5*absy3 - 2.5*absy2 + 1) * (absy <= 1) + (-0.5*absy3 + 2.5*absy2 - 4*absy + 2) * ((1 < absy) && (absy <= 2));    
        }
        kx_val = kernel_scale * kx_val;
        ky_val = kernel_scale * ky_val;
        float q = getP(in, y, x, in_height, in_width);

        val = val + ky_val * kx_val * q;
      }
    }

    out[out_idx] = val;
  }
}


__global__ void icgResampleZero(float* data, int length) {
  CUDA_KERNEL_LOOP(idx, length) {
    data[idx] = 0;
  }
}


template <int inter_kernel>
__global__ void icgResampleBackward(const float* out, float* in, int out_length, 
    int in_height, int in_width, int out_height, int out_width, float height_factor, float width_factor,
    int kernel_width, float kernel_scale) {

  long kernel_width_half = ceil(kernel_width / 2.0);

  CUDA_KERNEL_LOOP(out_idx, out_length) {
    float out_val = out[out_idx];

    float yo = (out_idx / out_width) + 0.5;
    float xo = (out_idx % out_width) + 0.5;

    float yi = yo / height_factor;
    float xi = xo / width_factor;

    int kh;
    for(kh = -kernel_width_half+1; kh <= kernel_width_half; ++kh) {
      int kw;
      for(kw = -kernel_width_half+1; kw <= kernel_width_half; ++kw) {
        float y = round(yi) + kh - 0.5;
        float x = round(xi) + kw - 0.5;
        
        float xdiff = kernel_scale * (xi - x);
        float ydiff = kernel_scale * (yi - y);

        float kx_val, ky_val;
        if(inter_kernel == 0) {
          kx_val = (-0.5 <= xdiff) && (xdiff < 0.5);
          ky_val = (-0.5 <= ydiff) && (ydiff < 0.5);
        }
        if(inter_kernel == 1) {
          kx_val = (xdiff+1) * ((-1 <= xdiff) && (xdiff < 0)) + (1-xdiff) * ((0 <= xdiff) & (xdiff <= 1));
          ky_val = (ydiff+1) * ((-1 <= ydiff) && (ydiff < 0)) + (1-ydiff) * ((0 <= ydiff) & (ydiff <= 1));
        }
        if(inter_kernel == 2) {
          float absx = fabs(xdiff);
          float absx2 = absx*absx;
          float absx3 = absx2*absx;
          float absy = fabs(ydiff);
          float absy2 = absy*absy;
          float absy3 = absy2*absy;
          kx_val = (1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) + (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) && (absx <= 2));
          ky_val = (1.5*absy3 - 2.5*absy2 + 1) * (absy <= 1) + (-0.5*absy3 + 2.5*absy2 - 4*absy + 2) * ((1 < absy) && (absy <= 2));    
        }
        kx_val = kernel_scale * kx_val;
        ky_val = kernel_scale * ky_val;
        
        int x_ = clampCoord(x, in_width);
        int y_ = clampCoord(y, in_height);
        int in_idx = y_ * in_width + x_;

        atomicAdd(in + in_idx, ky_val * kx_val * out_val);
      }
    }
  }
}



static int icgcunn_IcgResample_updateOutput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor* in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  const char* inter_method = luaT_getfieldcheckstring(L, 1, "inter_method");
  float height_factor = luaT_getfieldchecknumber(L, 1, "height_factor");
  float width_factor = luaT_getfieldchecknumber(L, 1, "width_factor");
  int antialiasing = luaT_getfieldcheckboolean(L, 1, "antialiasing");
  THCudaTensor* out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  int n_dim = in->nDimension;
  luaL_argcheck(L, n_dim == 3 || n_dim == 4, 2, "3D or 4D(batch mode) tensor expected");

  int num, channels, height, width, out_height, out_width;
  if(n_dim == 3) {
    num = 1;
    channels = THCudaTensor_size(state, in, 0);
    height = THCudaTensor_size(state, in, 1);
    width = THCudaTensor_size(state, in, 2);
    out_height = round(height * height_factor);
    out_width = round(width * width_factor);
    THCudaTensor_resize3d(state, out, channels, out_height, out_width);
  }
  else if(n_dim == 4) {
    num = THCudaTensor_size(state, in, 0);
    channels = THCudaTensor_size(state, in, 1);
    height = THCudaTensor_size(state, in, 2);
    width = THCudaTensor_size(state, in, 3);
    out_height = round(height * height_factor);
    out_width = round(width * width_factor);
    THCudaTensor_resize4d(state, out, num, channels, out_height, out_width);
  }

  int out_length = out_height * out_width;
  int in_length = height * width;
  in = THCudaTensor_newContiguous(state, in);

  // do resampling
  float* in_data = THCudaTensor_data(state, in);
  float* out_data = THCudaTensor_data(state, out);

  if(strcmp(inter_method, "nearest") == 0) {
    int kernel_width = 1;
    float kernel_scale = 1.0;

    if(antialiasing && height_factor < 1 && height_factor == width_factor) {
      kernel_width = (kernel_width / height_factor) + 1;
      kernel_scale = height_factor;
    }
    for(long n = 0; n < num; ++n) {
      for(long c = 0; c < channels; ++c) {
        icgResampleForward<0><<<GET_BLOCKS(out_length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          out_data, in_data, out_length, height, width, out_height, out_width, height_factor, width_factor, 
          kernel_width, kernel_scale);
        out_data = out_data + out_length;
        in_data = in_data + in_length;
      }
    }
  }
  else if(strcmp(inter_method, "bilinear") == 0) {
    int kernel_width = 2;
    float kernel_scale = 1.0;

    if(antialiasing && height_factor < 1 && height_factor == width_factor) {
      kernel_width = (kernel_width / height_factor) + 1;
      kernel_scale = height_factor;
    }
    for(long n = 0; n < num; ++n) {
      for(long c = 0; c < channels; ++c) {
        icgResampleForward<1><<<GET_BLOCKS(out_length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          out_data, in_data, out_length, height, width, out_height, out_width, height_factor, width_factor, 
          kernel_width, kernel_scale);
        out_data = out_data + out_length;
        in_data = in_data + in_length;
      }
    }
  }
  else if(strcmp(inter_method, "bicubic") == 0) {
    int kernel_width = 4;
    float kernel_scale = 1.0;

    if(antialiasing && height_factor < 1 && height_factor == width_factor) {
      kernel_width = (kernel_width / height_factor) + 1;
      kernel_scale = height_factor;
    }
    for(long n = 0; n < num; ++n) {
      for(long c = 0; c < channels; ++c) {
        icgResampleForward<2><<<GET_BLOCKS(out_length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          out_data, in_data, out_length, height, width, out_height, out_width, height_factor, width_factor, 
          kernel_width, kernel_scale);
        out_data = out_data + out_length;
        in_data = in_data + in_length;
      }
    }
  }
  else {
    luaL_error(L, "unknown inter method %s", inter_method);
  }

  THCudaTensor_free(state, in);

  THCudaCheck(cudaGetLastError());
  return 1;
}



static int icgcunn_IcgResample_updateGradInput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor* in = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* grad_out = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor* out = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  const char* inter_method = luaT_getfieldcheckstring(L, 1, "inter_method");
  float height_factor = luaT_getfieldchecknumber(L, 1, "height_factor");
  float width_factor = luaT_getfieldchecknumber(L, 1, "width_factor");
  int antialiasing = luaT_getfieldcheckboolean(L, 1, "antialiasing");
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
  long out_height = round(height * height_factor);
  long out_width = round(width * width_factor);
  
  THCudaTensor_resizeAs(state, grad_in, in);

  long out_length = out_height * out_width;
  long in_length = height * width;
  
  float* grad_out_data = THCudaTensor_data(state, grad_out);  
  float* grad_in_data = THCudaTensor_data(state, grad_in);

  float* grad_in_data_clr = grad_in_data;
  for(long n = 0; n < num; ++n) {
    for(long c = 0; c < channels; ++c) {
      icgResampleZero<<<GET_BLOCKS(in_length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        grad_in_data_clr, in_length);
      grad_in_data_clr = grad_in_data_clr + in_length;
    }
  }

  if(strcmp(inter_method, "nearest") == 0) {
    int kernel_width = 1;
    float kernel_scale = 1.0;

    if(antialiasing && height_factor < 1 && height_factor == width_factor) {
      kernel_width = (kernel_width / height_factor) + 1;
      kernel_scale = height_factor;
    }
    for(long n = 0; n < num; ++n) {
      for(long c = 0; c < channels; ++c) {
        icgResampleBackward<0><<<GET_BLOCKS(out_length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          grad_out_data, grad_in_data, out_length, height, width, out_height, out_width, height_factor, width_factor, 
          kernel_width, kernel_scale);
        grad_out_data = grad_out_data + out_length;
        grad_in_data = grad_in_data + in_length;
      }
    }
  }
  else if(strcmp(inter_method, "bilinear") == 0) {
    int kernel_width = 2;
    float kernel_scale = 1.0;

    if(antialiasing && height_factor < 1 && height_factor == width_factor) {
      kernel_width = (kernel_width / height_factor) + 1;
      kernel_scale = height_factor;
    }
    for(long n = 0; n < num; ++n) {
      for(long c = 0; c < channels; ++c) {
        icgResampleBackward<1><<<GET_BLOCKS(out_length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          grad_out_data, grad_in_data, out_length, height, width, out_height, out_width, height_factor, width_factor, 
          kernel_width, kernel_scale);
        grad_out_data = grad_out_data + out_length;
        grad_in_data = grad_in_data + in_length;
      }
    }
  }
  else if(strcmp(inter_method, "bicubic") == 0) {
    int kernel_width = 4;
    float kernel_scale = 1.0;

    if(antialiasing && height_factor < 1 && height_factor == width_factor) {
      kernel_width = (kernel_width / height_factor) + 1;
      kernel_scale = height_factor;
    }
    for(long n = 0; n < num; ++n) {
      for(long c = 0; c < channels; ++c) {
        icgResampleBackward<2><<<GET_BLOCKS(out_length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          grad_out_data, grad_in_data, out_length, height, width, out_height, out_width, height_factor, width_factor, 
          kernel_width, kernel_scale);
        grad_out_data = grad_out_data + out_length;
        grad_in_data = grad_in_data + in_length;
      }
    }
  }
  else {
    luaL_error(L, "unknown inter method %s", inter_method);
  }


  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg icgcunn_IcgResample__ [] = {
  {"IcgResample_updateOutput", icgcunn_IcgResample_updateOutput},
  {"IcgResample_updateGradInput", icgcunn_IcgResample_updateGradInput},
  {NULL, NULL}
};

void icgcunn_IcgResample_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, icgcunn_IcgResample__, "icgnn");
  lua_pop(L,1);
}

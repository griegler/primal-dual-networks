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



#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/IcgResample.c"
#else

#include "cpu.h"


static int icgnn_(IcgResample_updateOutput)(lua_State *L) {
  THTensor* in = luaT_checkudata(L, 2, torch_Tensor);

  const char* inter_method = luaT_getfieldcheckstring(L, 1, "inter_method");
  real height_factor = luaT_getfieldchecknumber(L, 1, "height_factor");
  real width_factor = luaT_getfieldchecknumber(L, 1, "width_factor");
  int antialiasing = luaT_getfieldcheckboolean(L, 1, "antialiasing");
  THTensor* out = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  long n_dim = in->nDimension;
  luaL_argcheck(L, n_dim == 3 || n_dim == 4, 2, "3D or 4D(batch mode) tensor expected");
    
  in = THTensor_(newContiguous)(in);

  long num, channels, height, width, out_height, out_width;
  if(n_dim == 3) {
    num = 1;
    channels = in->size[0];
    height = in->size[1];
    width = in->size[2];
    out_height = round(height * height_factor);
    out_width = round(width * width_factor);
    THTensor_(resize3d)(out, channels, out_height, out_width);
  }
  else if(n_dim == 4) {
    num = in->size[0];
    channels = in->size[1];
    height = in->size[2];
    width = in->size[3];
    out_height = round(height * height_factor);
    out_width = round(width * width_factor);
    THTensor_(resize4d)(out, num, channels, out_height, out_width);  
  }

  //do resampling
  real* in_data = THTensor_(data)(in);
  real* out_data = THTensor_(data)(out);


  TEMPLATE(rv_tensor, real) out_t = TEMPLATE(rv_tensor_create, real)(out_data, num, channels, height, width);
  TEMPLATE(rv_tensor, real) in_t = TEMPLATE(rv_tensor_create, real)(in_data, num, channels, height, width);

  real (*inter_kernel)(real);
  long kernel_width;
  real kernel_scale = 1.0;
  if(strcmp(inter_method, "nearest") == 0) {
    inter_kernel = &TEMPLATE(rv_box_fwd, real);
    kernel_width = 1;
  }
  else if(strcmp(inter_method, "bilinear") == 0) {
    inter_kernel = &TEMPLATE(rv_triangle_fwd, real);
    kernel_width = 2;
  }
  else if(strcmp(inter_method, "bicubic") == 0) {
    inter_kernel = &TEMPLATE(rv_cubic_fwd, real);
    kernel_width = 4;
  }
  else {
    luaL_error(L, "unknown inter method %s", inter_method);
  }

  if(antialiasing && height_factor < 1 && height_factor == width_factor) {
    kernel_width = (kernel_width / height_factor) + 1;
    kernel_scale = height_factor;
  }

  TEMPLATE(rv_resample_fwd, real)(out_t, in_t, height_factor, width_factor, 
      inter_kernel, kernel_width, kernel_scale);
  
  THTensor_(free)(in);

  return 1;
}



static int icgnn_(IcgResample_updateGradInput)(lua_State *L) {
  THTensor* in = luaT_checkudata(L, 2, torch_Tensor);
  THTensor* grad_out = luaT_checkudata(L, 3, torch_Tensor);

  const char* inter_method = luaT_getfieldcheckstring(L, 1, "inter_method");
  real height_factor = luaT_getfieldchecknumber(L, 1, "height_factor");
  real width_factor = luaT_getfieldchecknumber(L, 1, "width_factor");
  int antialiasing = luaT_getfieldcheckboolean(L, 1, "antialiasing");
  THTensor* out = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor* grad_in = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(grad_in, in);
  
  real* in_data = THTensor_(data)(in);
  real* out_data = THTensor_(data)(out);
  real* grad_out_data = THTensor_(data)(grad_out);
  real* grad_in_data = THTensor_(data)(grad_in);
 
  long n_dim = in->nDimension;
  long num, channels, height, width, out_height, out_width;
  if(n_dim == 3) {
    num = 1;
    channels = in->size[0];
    height = in->size[1];
    width = in->size[2];
  }
  else if(n_dim == 4) {
    num = in->size[0];
    channels = in->size[1];
    height = in->size[2];
    width = in->size[3];
  }

  
  TEMPLATE(rv_tensor, real) grad_in_t = TEMPLATE(rv_tensor_create, real)(grad_in_data, num, channels, height, width);
  TEMPLATE(rv_tensor, real) in_t = TEMPLATE(rv_tensor_create, real)(in_data, num, channels, height, width);
  TEMPLATE(rv_tensor, real) grad_out_t = TEMPLATE(rv_tensor_create, real)(grad_out_data, num, channels, height, width);

  real (*inter_kernel)(real);
  long kernel_width;
  real kernel_scale = 1.0;
  if(strcmp(inter_method, "nearest") == 0) {
    inter_kernel = &TEMPLATE(rv_box_fwd, real);
    kernel_width = 1;
  }
  else if(strcmp(inter_method, "bilinear") == 0) {
    inter_kernel = &TEMPLATE(rv_triangle_fwd, real);
    kernel_width = 2;
  }
  else if(strcmp(inter_method, "bicubic") == 0) {
    inter_kernel = &TEMPLATE(rv_cubic_fwd, real);
    kernel_width = 4;
  }
  else {
    luaL_error(L, "unknown inter method %s", inter_method);
  }

  if(antialiasing && height_factor < 1 && height_factor == width_factor) {
    kernel_width = (kernel_width / height_factor) + 1;
    kernel_scale = height_factor;
  }

  TEMPLATE(rv_resample_bwd, real)(grad_out_t, grad_in_t, height_factor, width_factor, 
      inter_kernel, kernel_width, kernel_scale);

  
  return 1;
}

static const struct luaL_Reg icgnn_(IcgResample__) [] = {
  {"IcgResample_updateOutput", icgnn_(IcgResample_updateOutput)},
  {"IcgResample_updateGradInput", icgnn_(IcgResample_updateGradInput)},
  {NULL, NULL}
};

static void icgnn_(IcgResample_init)(lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, icgnn_(IcgResample__), "icgnn");
  lua_pop(L,1);
}

#endif

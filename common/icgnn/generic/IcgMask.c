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
#define TH_GENERIC_FILE "generic/IcgMask.c"
#else


static void icgnn_(IcgMask_updateOutputGrid)(lua_State* L, THTensor* in, 
    THTensor* out, real mask_value) {

  int height_factor = luaT_getfieldcheckint(L, 1, "height_factor");
  int width_factor = luaT_getfieldcheckint(L, 1, "width_factor");    

  long n_dim = in->nDimension;
  luaL_argcheck(L, n_dim == 3 || n_dim == 4, 2, "3D or 4D(batch mode) tensor expected");
  in = THTensor_(newContiguous)(in);

  long num, channels, height, width;
  if(n_dim == 3) {
    num = 1;
    channels = in->size[0];
    height = in->size[1];
    width = in->size[2];
    THTensor_(resize3d)(out, channels, height, width);
  }
  else if(n_dim == 4) {
    num = in->size[0];
    channels = in->size[1];
    height = in->size[2];
    width = in->size[3];
    THTensor_(resize4d)(out, num, channels, height, width);  
  }

  real* in_data = THTensor_(data)(in);
  real* out_data = THTensor_(data)(out);

  long h_offset = (height_factor - 1) / 2;
  long w_offset = (width_factor - 1) / 2;

  long n;
#pragma omp parallel for private(n)
  for(n = 0; n < num * channels; ++n) {
    long h;
    for(h = 0; h < height; ++h) {
      long w;
      for(w = 0; w < width; ++w) {
        long idx = (n * height + h) * width + w;

        if(h % height_factor == h_offset && w % width_factor == w_offset) {
          out_data[idx] = in_data[idx];
        }
        else {
          out_data[idx] = mask_value;
        }
      }
    }  
  }
  
  THTensor_(free)(in);
}

static void icgnn_(IcgMask_updateOutputBorder)(lua_State* L, THTensor* in, 
    THTensor* out, real mask_value) {

  int border = luaT_getfieldcheckint(L, 1, "border");    

  long n_dim = in->nDimension;
  luaL_argcheck(L, n_dim == 3 || n_dim == 4, 2, "3D or 4D(batch mode) tensor expected");
  in = THTensor_(newContiguous)(in);

  long num, channels, height, width;
  if(n_dim == 3) {
    num = 1;
    channels = in->size[0];
    height = in->size[1];
    width = in->size[2];
    THTensor_(resize3d)(out, channels, height, width);
  }
  else if(n_dim == 4) {
    num = in->size[0];
    channels = in->size[1];
    height = in->size[2];
    width = in->size[3];
    THTensor_(resize4d)(out, num, channels, height, width);  
  }

  real* in_data = THTensor_(data)(in);
  real* out_data = THTensor_(data)(out);

  long n;
#pragma omp parallel for private(n)
  for(n = 0; n < num * channels; ++n) {
    long h;
    for(h = 0; h < height; ++h) {
      long w;
      for(w = 0; w < width; ++w) {
        long idx = (n * height + h) * width + w;

        if(h >= border && h < height - border && w >= border && w < width - border) {
          out_data[idx] = in_data[idx];
        }
        else {
          out_data[idx] = mask_value;
        }
      }
    }  
  }
  
  THTensor_(free)(in);
}


static int icgnn_(IcgMask_updateOutput)(lua_State* L) {
  THTensor* in = luaT_checkudata(L, 2, torch_Tensor);
  THTensor* out = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  const char* mask_type = luaT_getfieldcheckstring(L, 1, "mask_type");
  real mask_value = luaT_getfieldchecknumber(L, 1, "mask_value");

  if(strcmp(mask_type, "grid") == 0) {
    icgnn_(IcgMask_updateOutputGrid)(L, in, out, mask_value);
  }
  else if(strcmp(mask_type, "border") == 0) {
    icgnn_(IcgMask_updateOutputBorder)(L, in, out, mask_value);
  }
  else {
    luaL_error(L, "unknown mask type: %s", mask_type);
  }
  
  return 1;
}



static void icgnn_(IcgMask_updateGradInputGrid)(lua_State* L, THTensor* in,
    THTensor* grad_out, THTensor* out, THTensor* grad_in, real mask_value) {

  int height_factor = luaT_getfieldcheckint(L, 1, "height_factor");
  int width_factor = luaT_getfieldcheckint(L, 1, "width_factor");

  real* in_data = THTensor_(data)(in);
  real* out_data = THTensor_(data)(out);
  real* grad_in_data = THTensor_(data)(grad_in);
  real* grad_out_data = THTensor_(data)(grad_out);
 
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
  
  long h_offset = (height_factor - 1) / 2;
  long w_offset = (width_factor - 1) / 2;

  long n;
#pragma omp parallel for private(n)
  for(n = 0; n < num * channels; ++n) {
    long h;
    for(h = 0; h < height; ++h) {
      long w;
      for(w = 0; w < width; ++w) {
        long idx = (n * height + h) * width + w;

        if(h % height_factor == h_offset && w % width_factor == w_offset) {
          grad_in_data[idx] = grad_out_data[idx];
        }
        else {
          grad_in_data[idx] = 0;
        }
      }
    }
  }
}

static void icgnn_(IcgMask_updateGradInputBorder)(lua_State* L, THTensor* in,
    THTensor* grad_out, THTensor* out, THTensor* grad_in, real mask_value) {

  int border = luaT_getfieldcheckint(L, 1, "border");

  real* in_data = THTensor_(data)(in);
  real* out_data = THTensor_(data)(out);
  real* grad_in_data = THTensor_(data)(grad_in);
  real* grad_out_data = THTensor_(data)(grad_out);
 
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
  
  long n;
#pragma omp parallel for private(n)
  for(n = 0; n < num * channels; ++n) {
    long h;
    for(h = 0; h < height; ++h) {
      long w;
      for(w = 0; w < width; ++w) {
        long idx = (n * height + h) * width + w;

        if(h >= border && h < height - border && w >= border && w < width - border) {
          grad_in_data[idx] = grad_out_data[idx];
        }
        else {
          grad_in_data[idx] = 0;
        }
      }
    }
  }
}

static int icgnn_(IcgMask_updateGradInput)(lua_State *L) {
  THTensor* in = luaT_checkudata(L, 2, torch_Tensor);
  THTensor* grad_out = luaT_checkudata(L, 3, torch_Tensor);
  THTensor* out = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor* grad_in = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  
  THTensor_(resizeAs)(grad_in, in);

  const char* mask_type = luaT_getfieldcheckstring(L, 1, "mask_type");
  real mask_value = luaT_getfieldchecknumber(L, 1, "mask_value");

  if(strcmp(mask_type, "grid") == 0) {
    icgnn_(IcgMask_updateGradInputGrid)(L, in, grad_out, out, grad_in, mask_value);
  }
  else if(strcmp(mask_type, "border") == 0) {
    icgnn_(IcgMask_updateGradInputBorder)(L, in, grad_out, out, grad_in, mask_value);
  }
  else {
    luaL_error(L, "unknown mask type: %s", mask_type);
  }

  
  return 1;
}

static const struct luaL_Reg icgnn_(IcgMask__) [] = {
  {"IcgMask_updateOutput", icgnn_(IcgMask_updateOutput)},
  {"IcgMask_updateGradInput", icgnn_(IcgMask_updateGradInput)},
  {NULL, NULL}
};

static void icgnn_(IcgMask_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, icgnn_(IcgMask__), "icgnn");
  lua_pop(L,1);
}

#endif

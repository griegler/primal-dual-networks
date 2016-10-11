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
#define TH_GENERIC_FILE "generic/IcgGeneralizedNabla.c"
#else

static int icgnn_(IcgGeneralizedNabla_updateOutput)(lua_State *L) {
  THTensor* in = luaT_checkudata(L, 2, torch_Tensor);
  THTensor* dir = luaT_getfieldcheckudata(L, 1, "directions", torch_Tensor);
  THTensor* out = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  long n_dim = in->nDimension;
  luaL_argcheck(L, n_dim == 3 || n_dim == 4, 2, "3D or 4D(batch mode) tensor expected");
    
  in = THTensor_(newContiguous)(in);
  real* in_data = THTensor_(data)(in);

  real* dir_data = THTensor_(data)(dir);
  long n_dir = dir->size[1];

  long num, channels, height, width;
  if(n_dim == 3) {
    num = 1;
    channels = in->size[0];
    height = in->size[1];
    width = in->size[2];
    THTensor_(resize3d)(out, n_dir * channels, height, width);
  }
  else if(n_dim == 4) {
    num = in->size[0];
    channels = in->size[1];
    height = in->size[2];
    width = in->size[3];
    THTensor_(resize4d)(out, num, n_dir * channels, height, width);  
  }

  real* out_data = THTensor_(data)(out);

  long offset = height * width;
  long n;
#pragma omp parallel for private(n)
  for(n = 0; n < num * channels; ++n) {
    long h;
    for(h = 0; h < height; ++h) {
      long w;
      for(w = 0; w < width; ++w) {
        long in_idx = (n * height + h) * width + w;
        
        int dir_idx = 0;
        for(dir_idx = 0; dir_idx < n_dir; ++dir_idx) {
          long dir_x = dir_data[dir_idx];
          long dir_y = dir_data[n_dir + dir_idx];

          long out_idx = ((n * n_dir + dir_idx) * height + h) * width + w;

          if(w >= -dir_x && w < width - dir_x && h >= -dir_y && h < height - dir_y) {
            out_data[out_idx] = in_data[in_idx + dir_y * width + dir_x] - in_data[in_idx];
          }
          else {
            out_data[out_idx] = 0;
          }
          // printf("  @(h=%d, w=%d) out=%f | dir_x=%d, dir_y=%d, dir_idx=%d\n",
          //     h, w, out_data[out_idx], dir_x, dir_y, dir_idx);
        }

      }
    }  
  }
  
  THTensor_(free)(in);

  return 1;
}

static int icgnn_(IcgGeneralizedNabla_updateGradInput)(lua_State *L) {
  THTensor *in = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *grad_out = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *out = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor* dir = luaT_getfieldcheckudata(L, 1, "directions", torch_Tensor);
  THTensor *grad_in = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(grad_in, in);
  
  real* in_data = THTensor_(data)(in);
  real* out_data = THTensor_(data)(out);
  real* grad_in_data = THTensor_(data)(grad_in);
  real* grad_out_data = THTensor_(data)(grad_out);

  real* dir_data = THTensor_(data)(dir);
  long n_dir = dir->size[1];
 
  long n_dim = in->nDimension;
  long num, channels, height, width;
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

  long offset = height * width;
  long n;
#pragma omp parallel for private(n)
  for(n = 0; n < num * channels; ++n) {
    long h;
    for(h = 0; h < height; ++h) {
      long w;
      for(w = 0; w < width; ++w) {
        long in_idx = (n * height + h) * width + w;
        
        real grad = 0;
        int dir_idx = 0;
        for(dir_idx = 0; dir_idx < n_dir; ++dir_idx) {
          long dir_x = dir_data[dir_idx];
          long dir_y = dir_data[n_dir + dir_idx];

          long out_idx = ((n * n_dir + dir_idx) * height + h) * width + w;
          // printf("out_idx=%d | n=%d, n_dir=%d, dir_idx=%d, height=%d, h=%d, width=%d, w=%d\n",
          //     out_idx, n, n_dir, dir_idx, height, h, width, w);

          if(w >= dir_x && w < width+dir_x && h >= dir_y && h < height+dir_y) {
            grad = grad + (+1) * grad_out_data[out_idx - dir_y * width - dir_x];
            // printf("  @(h=%d, w=%d), +%f @(idx=%d, out_idx=%d, dir_y=%d, dir_x=%d)\n", 
            //     h, w, grad_out_data[out_idx - dir_y * width - dir_x], out_idx - dir_y * width - dir_x, out_idx, dir_y, dir_x);
          }
          if(w >= -dir_x && w < width-dir_x && h >= -dir_y && h < height-dir_y) {
            grad = grad + (-1) * grad_out_data[out_idx];
            // printf("  @(h=%d, w=%d), -%f @(idx=%d)\n", h, w, grad_out_data[out_idx], out_idx);
          }
        }
        grad_in_data[in_idx] = grad;
      }
    }
  }
  
  return 1;
}

static const struct luaL_Reg icgnn_(IcgGeneralizedNabla__) [] = {
  {"IcgGeneralizedNabla_updateOutput", icgnn_(IcgGeneralizedNabla_updateOutput)},
  {"IcgGeneralizedNabla_updateGradInput", icgnn_(IcgGeneralizedNabla_updateGradInput)},
  {NULL, NULL}
};

static void icgnn_(IcgGeneralizedNabla_init)(lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, icgnn_(IcgGeneralizedNabla__), "icgnn");
  lua_pop(L,1);
}

#endif

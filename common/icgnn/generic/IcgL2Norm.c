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
#define TH_GENERIC_FILE "generic/IcgL2Norm.c"
#else

static int icgnn_(IcgL2Norm_updateOutput)(lua_State *L) {
  THTensor* in = luaT_checkudata(L, 2, torch_Tensor);
  int spatial = luaT_getfieldcheckboolean(L, 1, "spatial");
  THTensor* out = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  long n_dim = in->nDimension;
  luaL_argcheck(L, n_dim == 3 || n_dim == 4, 2, "3D or 4D(batch mode) tensor expected");
    
  in = THTensor_(newContiguous)(in);
  real* in_data = THTensor_(data)(in);
  
  long num, channels, height, width, out_channels;
  if(n_dim == 3) {
    num = 1;
    channels = in->size[0];
    height = in->size[1];
    width = in->size[2];
    out_channels = spatial ? channels : 1;
    THTensor_(resize3d)(out, out_channels, height, width);
  }
  else if(n_dim == 4) {
    num = in->size[0];
    channels = in->size[1];
    height = in->size[2];
    width = in->size[3];
    out_channels = spatial ? channels : 1;
    THTensor_(resize4d)(out, num, out_channels, height, width);  
  }
  real* out_data = THTensor_(data)(out);

  long n=0;
#pragma omp parallel for private(n)
  for(n = 0; n < num; ++n) {
    long h;
    for(h = 0; h < height; ++h) {
      long w;
      for(w = 0; w < width; ++w) {
        long c;
        float out_val = 0;
        for(c = 0; c < channels; ++c) {
          long idx = ((n * channels + c) * height + h) * width + w;
          real in_val = in_data[idx];
          out_val = out_val + in_val * in_val;
        }
        out_val = sqrt(out_val);

        for(c = 0; c < out_channels; ++c) {
          long idx = ((n * out_channels + c) * height + h) * width + w;
          out_data[idx] = out_val; 
        }
      }
    }
  }
  
  THTensor_(free)(in);

  return 1;
}

static int icgnn_(IcgL2Norm_updateGradInput)(lua_State *L) {
  THTensor *in = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *grad_out = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *out = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int spatial = luaT_getfieldcheckboolean(L, 1, "spatial");
  real dft = luaT_getfieldchecknumber(L, 1, "dft");
  THTensor *grad_in = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(grad_in, in);
  
  real* in_data = THTensor_(data)(in);
  real* out_data = THTensor_(data)(out);
  real* grad_in_data = THTensor_(data)(grad_in);
  real* grad_out_data = THTensor_(data)(grad_out);

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
  long out_channels = spatial ? channels : 1;

  long n;
#pragma omp parallel for private(n)
  for(n = 0; n < num; ++n) {
    long h;
    for(h = 0; h < height; ++h) {
      long w;
      for(w = 0; w < width; ++w) {
        long ci;
        // float out_val = 0;
        // for(ci = 0; ci < channels; ++ci) {
        //   long idx = ((n * channels + ci) * height + h) * width + w;
        //   real in_val = in_data[idx];
        //   out_val = out_val + in_val * in_val;
        // }
        // out_val = sqrt(out_val);
        // out_val = out_val == 0 ? dft : out_val;

        for(ci = 0; ci < channels; ++ci) {
          long in_idx = ((n * channels + ci) * height + h) * width + w;
          float grad_in_val = 0;

          // printf("-------- %d/%d|%d/%d/%d --------\n", num, channels, out_channels, height, width);
          long co;
          for(co = 0; co < out_channels; ++co) {
            long out_idx = ((n * out_channels + co) * height + h) * width + w;
            real out_val = out_data[out_idx];
            out_val = out_val == 0 ? dft : out_val;
            grad_in_val = grad_in_val + (grad_out_data[out_idx] * in_data[in_idx] / out_val);

            // printf("  @%d/%d|%d/%d/%d - %d/%d: %f * %f / %f \n", 
            //     n, ci, co, h, w, in_idx, out_idx, grad_out_data[out_idx], in_data[in_idx], out_val);
          }
          grad_in_data[in_idx] = grad_in_val;
          // printf("    = %f\n", grad_in_val);
        }

      }
    }
  }
  
  return 1;
}

static const struct luaL_Reg icgnn_(IcgL2Norm__) [] = {
  {"IcgL2Norm_updateOutput", icgnn_(IcgL2Norm_updateOutput)},
  {"IcgL2Norm_updateGradInput", icgnn_(IcgL2Norm_updateGradInput)},
  {NULL, NULL}
};

static void icgnn_(IcgL2Norm_init)(lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, icgnn_(IcgL2Norm__), "icgnn");
  lua_pop(L,1);
}

#endif

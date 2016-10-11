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
#define TH_GENERIC_FILE "generic/IcgNoise.c"
#else

#include "THRandom.h"


static void icgnn_(IcgNoise_gaussian)(lua_State* L, THGenerator* rng, 
    THTensor* input, THTensor* output, int inplace) {
  
  real mu = luaT_getfieldchecknumber(L, 1, "mu");
  real sigma = luaT_getfieldchecknumber(L, 1, "sigma");

  if(inplace) {
    TH_TENSOR_APPLY(real, input, \
        *input_data = *input_data + THRandom_normal(rng, mu, sigma););
  }
  else {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(real, output, real, input, \
      *output_data = *input_data + THRandom_normal(rng, mu, sigma););
  }
}

static void icgnn_(IcgNoise_speckle)(lua_State* L, THGenerator* rng, 
    THTensor* input, THTensor* output, int inplace) {
  
  real mu = luaT_getfieldchecknumber(L, 1, "mu");
  real sigma = luaT_getfieldchecknumber(L, 1, "sigma");

  if(inplace) {
    TH_TENSOR_APPLY(real, input, \
        real val = *input_data; \
        *input_data = val + val * THRandom_normal(rng, mu, sigma););
  }
  else {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(real, output, real, input, \
      real val = *input_data; \
      *output_data = val + val * THRandom_normal(rng, mu, sigma););
  }
}

static void icgnn_(IcgNoise_localvar)(lua_State* L, THGenerator* rng, 
    THTensor* input, THTensor* output, int inplace) {
  
  real sigma = luaT_getfieldchecknumber(L, 1, "sigma");
  real k = luaT_getfieldchecknumber(L, 1, "k");
  int inverse = luaT_getfieldcheckboolean(L, 1, "inverse");

  if(inplace) {
    TH_TENSOR_APPLY(real, input, \
        real val = *input_data; \
        real s = inverse ? sigma / val : sigma * val; \
        real n = (s > 0 && val > 0) ? THRandom_normal(rng, 0, s) : 0; \
        *input_data = val + k * n;);
  }
  else {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(real, output, real, input, \
      real val = *input_data; \
      real s = inverse ? sigma / val : sigma * val; \
      real n = (s > 0 && val > 0) ? THRandom_normal(rng, 0, s) : 0; \
      *output_data = val + k * n;);
  }
}


static int icgnn_(IcgNoise_updateOutput)(lua_State *L)
{
  THTensor* input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor* output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int inplace = luaT_getfieldcheckboolean(L, 1, "inplace");

  const char* noise_type = luaT_getfieldcheckstring(L, 1, "noise_type");

  THGenerator* rng = THGenerator_new();
 
  if(!inplace) {
    THTensor_(resizeAs)(output, input);
  }

  if(strcmp(noise_type, "gaussian") == 0 || strcmp(noise_type, "normal") == 0) {
    icgnn_(IcgNoise_gaussian)(L, rng, input, output, inplace);
  }
  else if(strcmp(noise_type, "speckle") == 0) {
    icgnn_(IcgNoise_speckle)(L, rng, input, output, inplace);
  }
  else if(strcmp(noise_type, "localvar") == 0) {
    icgnn_(IcgNoise_localvar)(L, rng, input, output, inplace);
  }
  else {
    luaL_error(L, "unknown noise type %s", noise_type);
  }

  if(inplace) {
    THTensor_(set)(output, input);
  }

  THGenerator_free(rng);

  return 1;
}

static const struct luaL_Reg icgnn_(IcgNoise__) [] = {
  {"IcgNoise_updateOutput", icgnn_(IcgNoise_updateOutput)},
  {NULL, NULL}
};

static void icgnn_(IcgNoise_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, icgnn_(IcgNoise__), "icgnn");
  lua_pop(L,1);
}

#endif

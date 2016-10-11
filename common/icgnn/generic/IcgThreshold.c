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
#define TH_GENERIC_FILE "generic/IcgThreshold.c"
#else

static int icgnn_(IcgThreshold_updateOutput)(lua_State *L) {
  THTensor* input = luaT_checkudata(L, 2, torch_Tensor);
  real threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  real val_greater = luaT_getfieldchecknumber(L, 1, "val_greater");
  real val_smaller = luaT_getfieldchecknumber(L, 1, "val_smaller");
  THTensor* output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  input = THTensor_(newContiguous)(input);
  real* in = THTensor_(data)(input);
  
  THTensor_(resizeAs)(output, input);
  real* out = THTensor_(data)(output);

  long nelem = THTensor_(nElement)(input);
  long idx;
#pragma omp parallel for private(idx)
  for(idx = 0; idx < nelem; ++idx) {
    out[idx] = in[idx] > threshold ? val_greater : val_smaller;
  }
  
  THTensor_(free)(input);

  return 1;
}

static int icgnn_(IcgThreshold_updateGradInput)(lua_State *L) {
  THTensor* input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor* grad_input = luaT_checkudata(L, 3, torch_Tensor);
 
  THTensor_(resizeAs)(grad_input, input);
  
  real* grad_in1 = THTensor_(data)(grad_input);

  long nelem = THTensor_(nElement)(input);
  long idx;
#pragma omp parallel for private(idx)
  for(idx = 0; idx < nelem; ++idx) {
    grad_in1[idx] = 0;
  }
  
  return 1;
}

static const struct luaL_Reg icgnn_(IcgThreshold__) [] = {
  {"IcgThreshold_updateOutput", icgnn_(IcgThreshold_updateOutput)},
  {"IcgThreshold_updateGradInput", icgnn_(IcgThreshold_updateGradInput)},
  {NULL, NULL}
};

static void icgnn_(IcgThreshold_init)(lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, icgnn_(IcgThreshold__), "icgnn");
  lua_pop(L,1);
}

#endif

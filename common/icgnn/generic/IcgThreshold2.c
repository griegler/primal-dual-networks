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
#define TH_GENERIC_FILE "generic/IcgThreshold2.c"
#else

static int icgnn_(IcgThreshold2_updateOutput)(lua_State *L) {
  THTensor* input1 = luaT_checkudata(L, 2, torch_Tensor);
  THTensor* input2 = luaT_checkudata(L, 3, torch_Tensor);
  real threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  real val = luaT_getfieldchecknumber(L, 1, "val");
  THTensor* output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  input1 = THTensor_(newContiguous)(input1);
  input2 = THTensor_(newContiguous)(input2);
  real* in1 = THTensor_(data)(input1);
  real* in2 = THTensor_(data)(input2);
  
  THTensor_(resizeAs)(output, input1);
  real* out = THTensor_(data)(output);

  long nelem = THTensor_(nElement)(input1);
  luaL_argcheck(L, nelem == THTensor_(nElement)(input2), 2, "input1 should have same number of elements as input2");

  long idx;
#pragma omp parallel for private(idx)
  for(idx = 0; idx < nelem; ++idx) {
    out[idx] = in2[idx] <= threshold ? val : in1[idx];
  }
  
  THTensor_(free)(input1);
  THTensor_(free)(input2);

  return 1;
}

static int icgnn_(IcgThreshold2_updateGradInput)(lua_State *L) {
  THTensor* input1 = luaT_checkudata(L, 2, torch_Tensor);
  THTensor* input2 = luaT_checkudata(L, 3, torch_Tensor);
  THTensor* grad_input1 = luaT_checkudata(L, 4, torch_Tensor);
  THTensor* grad_output = luaT_checkudata(L, 5, torch_Tensor);
  real threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  real val = luaT_getfieldchecknumber(L, 1, "val");
 
  THTensor_(resizeAs)(grad_input1, input1);
  
  real* in1 = THTensor_(data)(input1);
  real* in2 = THTensor_(data)(input2);
  real* grad_in1 = THTensor_(data)(grad_input1);
  real* grad_out = THTensor_(data)(grad_output);

  long nelem = THTensor_(nElement)(input1);
  luaL_argcheck(L, nelem == THTensor_(nElement)(input2), 2, "input1 should have same number of elements as input2");

  long idx;
#pragma omp parallel for private(idx)
  for(idx = 0; idx < nelem; ++idx) {
    grad_in1[idx] = in2[idx] <= threshold ? 0 : grad_out[idx];
  }
  
  return 1;
}

static const struct luaL_Reg icgnn_(IcgThreshold2__) [] = {
  {"IcgThreshold2_updateOutput", icgnn_(IcgThreshold2_updateOutput)},
  {"IcgThreshold2_updateGradInput", icgnn_(IcgThreshold2_updateGradInput)},
  {NULL, NULL}
};

static void icgnn_(IcgThreshold2_init)(lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, icgnn_(IcgThreshold2__), "icgnn");
  lua_pop(L,1);
}

#endif

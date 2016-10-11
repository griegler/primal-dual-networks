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



#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define icgnn_(NAME) TH_CONCAT_3(icgnn_, Real, NAME)

#include "generic/IcgL2Norm.c"
#include "THGenerateFloatTypes.h"

#include "generic/IcgNabla.c"
#include "THGenerateFloatTypes.h"

#include "generic/IcgNablaT.c"
#include "THGenerateFloatTypes.h"

#include "generic/IcgGeneralizedNabla.c"
#include "THGenerateFloatTypes.h"

#include "generic/IcgGeneralizedNablaT.c"
#include "THGenerateFloatTypes.h"

#include "generic/IcgNoise.c"
#include "THGenerateFloatTypes.h"

#include "generic/IcgResample.c"
#include "THGenerateFloatTypes.h"

#include "generic/IcgMask.c"
#include "THGenerateFloatTypes.h"

#include "generic/IcgThreshold.c"
#include "THGenerateFloatTypes.h"

#include "generic/IcgThreshold2.c"
#include "THGenerateFloatTypes.h"

#include "generic/IcgThreshold3.c"
#include "THGenerateFloatTypes.h"


LUA_EXTERNC DLL_EXPORT int luaopen_libicgnn(lua_State *L);

int luaopen_libicgnn(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "icgnn");

  icgnn_FloatIcgL2Norm_init(L);
  icgnn_FloatIcgNabla_init(L);
  icgnn_FloatIcgNablaT_init(L);
  icgnn_FloatIcgGeneralizedNabla_init(L);
  icgnn_FloatIcgGeneralizedNablaT_init(L);
  icgnn_FloatIcgNoise_init(L);
  icgnn_FloatIcgResample_init(L);
  icgnn_FloatIcgMask_init(L);
  icgnn_FloatIcgThreshold_init(L);
  icgnn_FloatIcgThreshold2_init(L);
  icgnn_FloatIcgThreshold3_init(L);
  
  icgnn_DoubleIcgL2Norm_init(L);
  icgnn_DoubleIcgNabla_init(L);
  icgnn_DoubleIcgNablaT_init(L);
  icgnn_DoubleIcgGeneralizedNabla_init(L);
  icgnn_DoubleIcgGeneralizedNablaT_init(L);
  icgnn_DoubleIcgNoise_init(L);
  icgnn_DoubleIcgResample_init(L);
  icgnn_DoubleIcgMask_init(L);
  icgnn_DoubleIcgThreshold_init(L);
  icgnn_DoubleIcgThreshold2_init(L);
  icgnn_DoubleIcgThreshold3_init(L);

  return 1;
}


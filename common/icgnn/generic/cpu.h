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




#define CONCAT(X, Y) X##_##Y
#define TEMPLATE(X, Y) CONCAT(X, Y)


#if defined(T)
// Template functions start here

typedef struct {
  T* data;
  long num;
  long channels;
  long height;
  long width;
} TEMPLATE(rv_tensor, T);


//-------------------------------------------------------------------------------
// General functions
//-------------------------------------------------------------------------------
TEMPLATE(rv_tensor, T) TEMPLATE(rv_tensor_create, T)(T* data, long num, 
    long channels, long height, long width) {
  TEMPLATE(rv_tensor, T) t;
  t.data = data;
  t.num = num;
  t.channels = channels;
  t.height = height;
  t.width = width;
  return t;
}

long TEMPLATE(rv_clamp_coord, T)(long x, long size) {
  return x <= 0 ? 0 : (x >= size ? size - 1 : x);
}

long TEMPLATE(rv_get_idx, T)(TEMPLATE(rv_tensor, T) t, long n, long c, long h, long w) {
  return ((n * t.channels + c) * t.height + h) * t.width + w;
}

T TEMPLATE(rv_get_val, T)(TEMPLATE(rv_tensor, T) t, long n, long c, long h, long w) {
  return t.data[TEMPLATE(rv_get_idx, T)(t, n, c, h, w)];
}

T TEMPLATE(rv_get_val_hwbounds, T)(TEMPLATE(rv_tensor, T) t, long n, long c, long h, long w) {
  h = TEMPLATE(rv_clamp_coord, T)(h, t.height);
  w = TEMPLATE(rv_clamp_coord, T)(w, t.width);
  return t.data[TEMPLATE(rv_get_idx, T)(t, n, c, h, w)];
}

void TEMPLATE(rv_zero, T)(TEMPLATE(rv_tensor, T) t) {
  long idx = 0;
  int n_elem = t.num * t.channels * t.height * t.width;
  #pragma omp parallel for private(idx)
  for(idx = 0; idx < n_elem; ++idx) {
    t.data[idx] = 0;
  }
}

T TEMPLATE(rv_box_fwd, T)(T x) {
  return (-0.5 <= x) && (x < 0.5);
}

T TEMPLATE(rv_triangle_fwd, T)(T x) {
  return (x+1) * ((-1 <= x) && (x < 0)) + (1-x) * ((0 <= x) & (x <= 1));
}

T TEMPLATE(rv_cubic_fwd, T)(T x) {
  T absx = fabs(x);
  T absx2 = absx*absx;
  T absx3 = absx2*absx;
  return (1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) + 
         (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) && (absx <= 2));
}

T TEMPLATE(rv_box_bwd, T)(T x) {
  return 0;
}

T TEMPLATE(rv_triangle_bwd, T)(T x) {
  return (1) * ((-1 <= x) && (x < 0)) + (-1) * ((0 <= x) & (x <= 1));
}

T TEMPLATE(rv_cubic_bwd, T)(T x) {
  T absx = fabs(x);
  T sgn = (x > 0) - (x < 0);
  return (4.5 * x * absx - 5 * x) * (absx <= 1) + 
         (-1.5 * x * absx + 5 * x - 4 * sgn) * ((1 < absx) && (absx <= 2));
}


#include "cpu_resample.h"


#elif !defined(RV_NN_CPU_)
#define RV_NN_CPU_
// Template instantiations start here
#define T float
#include "cpu.h"
#undef T

#define T double
#include "cpu.h"
#undef T

#endif //RV_NN_CPU_

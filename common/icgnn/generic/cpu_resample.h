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





void TEMPLATE(rv_resample_fwd, T)(TEMPLATE(rv_tensor, T) out,
    const TEMPLATE(rv_tensor, T) in, T height_factor, T width_factor,
    T (*inter_kernel)(T), long kernel_width, T kernel_scale) {
  
  long in_height = in.height;
  long in_width = in.width;
  long num_channels = in.num * in.channels;

  long out_height = round(height_factor * in_height);
  long out_width = round(width_factor * in_width);

  long kernel_width_half = ceil(kernel_width / 2.0);

  long n;
#pragma omp parallel for private(n)
  for(n = 0; n < num_channels; ++n) {
    long h;
    for(h = 0; h < out_height; ++h) {
      long w;
      for(w = 0; w < out_width; ++w) {
        T yo = h + 0.5;
        T xo = w + 0.5;

        T yi = yo / height_factor;
        T xi = xo / width_factor;

        // printf("@ h=%d, w=%d | yi=%f, xi=%f | yo=%f, xo=%f | kernel_width=%d, kernel_width_half=%d\n", h, w, yi, xi, yo, xo, kernel_width, kernel_width_half);

        T val = 0;
        long kh;
        for(kh = -kernel_width_half+1; kh <= kernel_width_half; ++kh) {
          long kw;
          for(kw = -kernel_width_half+1; kw <= kernel_width_half; ++kw) {
            T y = round(yi) + kh - 0.5;
            T x = round(xi) + kw - 0.5;

            T kx_val = kernel_scale * (*inter_kernel)(kernel_scale * (xi - x));
            T ky_val = kernel_scale * (*inter_kernel)(kernel_scale * (yi - y));
            T q = TEMPLATE(rv_get_val_hwbounds, T)(in, 0, n, y, x);

            // printf("  @ kh=%d, kw=%d | y=%f, x=%f | kyval=%f, kxval=%f, q=%f\n", kh, kw, y, x, ky_val, kx_val, q);

            val = val + ky_val * kx_val * q;
          }
        }

        long out_idx = (n * out_height + h) * out_width + w;
        out.data[out_idx] = val;
      }
    }  
  }
}

void TEMPLATE(rv_resample_bwd, T)(TEMPLATE(rv_tensor, T) out, 
    const TEMPLATE(rv_tensor, T) in, T height_factor, T width_factor,
    T (*inter_kernel)(T), long kernel_width, T kernel_scale) {

  long in_height = in.height;
  long in_width = in.width;
  long num_channels = in.num * in.channels;

  long out_height = round(height_factor * in_height);
  long out_width = round(width_factor * in_width);

  long kernel_width_half = ceil(kernel_width / 2.0);

  TEMPLATE(rv_zero, T)(in);

  long n;
#pragma omp parallel for private(n)
  for(n = 0; n < num_channels; ++n) {
    long h;
    for(h = 0; h < out_height; ++h) {
      long w;
      for(w = 0; w < out_width; ++w) {
        long out_idx = (n * out_height + h) * out_width + w;
        T out_val = out.data[out_idx];

        T yo = h + 0.5;
        T xo = w + 0.5;

        T yi = yo / height_factor;
        T xi = xo / width_factor;

        long kh;
        for(kh = -kernel_width_half+1; kh <= kernel_width_half; ++kh) {
          long kw;
          for(kw = -kernel_width_half+1; kw <= kernel_width_half; ++kw) {
            T y = round(yi) + kh - 0.5;
            T x = round(xi) + kw - 0.5;

            T kx_val = kernel_scale * (*inter_kernel)(kernel_scale * (xi - x));
            T ky_val = kernel_scale * (*inter_kernel)(kernel_scale * (yi - y));
            
            long x_ = TEMPLATE(rv_clamp_coord, T)(x, in_width);
            long y_ = TEMPLATE(rv_clamp_coord, T)(y, in_height);
            long in_idx = (n * in_height + y_) * in_width + x_;

            in.data[in_idx] = in.data[in_idx] + ky_val * kx_val * out_val;
            // printf("  x_=%d, y_=%d, in_idx=%d | ky_val=%f, kx_val=%f, out_val=%f\n", x_, y_, in_idx, ky_val, kx_val, out_val);
          }
        }
      }
    }  
  }
}


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



#include "utils.h"
#include "common.h"


__host__ __device__ float icgCudaRenderInterLinePlane(const float* r0, 
    const float* rd, const float* p, float bg_val) {
  float x0 = r0[0];
  float y0 = r0[1];
  float z0 = r0[2];

  float xd = rd[0];
  float yd = rd[1];
  float zd = rd[2];

  float x1 = p[0];
  float y1 = p[1];
  float z1 = p[2];
  float x2 = p[3];
  float y2 = p[4];
  float z2 = p[5];
  float x3 = p[6];
  float y3 = p[7];
  float z3 = p[8];
  float x4 = p[9];
  float y4 = p[10];
  float z4 = p[11];

  float A = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2);
  float B = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2);
  float C = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);
  float D = -x1 * (y2*z3 - y3*z2) - x2 * (y3*z1 - y1*z3) - x3 * (y1*z2 - y2*z1);

  float t = -(A * x0 + B * y0 + C * z0 + D) / (A * xd + B * yd + C * zd);
  float xi = x0 + t * xd;
  float yi = y0 + t * yd;
  float zi = z0 + t * zd;

  float v1 = x1 * (x2 - x1) + y1 * (y2 - y1) + z1 * (z2 - z1);
  float v2 = xi * (x2 - x1) + yi * (y2 - y1) + zi * (z2 - z1);
  float v3 = x2 * (x2 - x1) + y2 * (y2 - y1) + z2 * (z2 - z1);
  float v4 = x1 * (x4 - x1) + y1 * (y4 - y1) + z1 * (z4 - z1);
  float v5 = xi * (x4 - x1) + yi * (y4 - y1) + zi * (z4 - z1);
  float v6 = x4 * (x4 - x1) + y4 * (y4 - y1) + z4 * (z4 - z1);

  bool within = (v1 <= v2 && v2 <= v3) && (v4 <= v5 && v5 <= v6);
  
  /* return within && t >= 0 ? t : bg_val; */
  return within ? (t < 0 ? 0 : t) : bg_val;
}

__host__ __device__ float icgCudaRenderInterLineSphere(const float* r0, 
    const float* rd, const float* s, float bg_val) {
  float x0 = r0[0];
  float y0 = r0[1];
  float z0 = r0[2];

  float rd_norm = sqrt(rd[0] * rd[0] + rd[1] * rd[1] + rd[2] * rd[2]);
  float xd = rd[0] / rd_norm;
  float yd = rd[1] / rd_norm;
  float zd = rd[2] / rd_norm;

  float xc = s[0];
  float yc = s[1];
  float zc = s[2];
  float r = s[3];

  float rd_dot_r0_m_c = xd * (x0 - xc) + yd * (y0 - yc) + zd * (z0 - zc);
  float r0_m_c_dot = (x0 - xc) * (x0 - xc) + (y0 - yc) * (y0 - yc) + (z0 - zc) * (z0 - zc);
  float radicand = rd_dot_r0_m_c * rd_dot_r0_m_c - r0_m_c_dot + r * r;

  float t1 = -rd_dot_r0_m_c + sqrt(radicand);
  float t2 = -rd_dot_r0_m_c - sqrt(radicand);

  return radicand < 0 ? bg_val : (t1 < t2 ? t1 : t2);
}

__global__ void icgCudaRenderRender(float* img, int length, int img_height, int img_width,
    float cam_x, float cam_y, float cam_z, float step_x, float step_y, 
    int n_planes, const float* planes, int n_spheres, const float* spheres, 
    float bg_val) {

  CUDA_KERNEL_LOOP(img_idx, length) {
    int h = img_height - img_idx / img_width - 1;
    int w = img_idx % img_width;

    float r0[3];
    r0[0] = cam_x + w * step_x; 
    r0[1] = cam_y + h * step_y;
    r0[2] = cam_z;
    float rd[3];
    rd[0] = 0;
    rd[1] = 0;
    rd[2] = 1;

    float val = bg_val;
    for(int pl_idx = 0; pl_idx < n_planes; ++pl_idx) {
      float pl_val = icgCudaRenderInterLinePlane(r0, rd, planes + 3 * 4 * pl_idx, bg_val);
      val = pl_val < val ? pl_val : val;
    }

    for(int sp_idx = 0; sp_idx < n_spheres; ++sp_idx) {
      float sp_val = icgCudaRenderInterLineSphere(r0, rd, spheres + 4 * sp_idx, bg_val);
      val = sp_val < val ? sp_val : val;
    }

    img[img_idx] = val < 0 ? 0 : val;
  }
}

static int icgrender_IcgCudaRender_render(lua_State *L)
{
  THCState* state = getCutorchState(L);
  THCudaTensor* planes = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* spheres = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

  double cam_x = luaT_getfieldchecknumber(L, 1, "cam_x");
  double cam_y = luaT_getfieldchecknumber(L, 1, "cam_y");
  double cam_z = luaT_getfieldchecknumber(L, 1, "cam_z");
  long img_height = luaT_getfieldchecknumber(L, 1, "img_height");
  long img_width = luaT_getfieldchecknumber(L, 1, "img_width");
  double step_x = luaT_getfieldchecknumber(L, 1, "step_x");
  double step_y = luaT_getfieldchecknumber(L, 1, "step_y");
  double bg_val = luaT_getfieldchecknumber(L, 1, "bg_val");
  THCudaTensor* img = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "img", "torch.CudaTensor");

  THCudaTensor_resize2d(state, img, img_height, img_width);

  long n_planes = 0;
  if(THCudaTensor_nDimension(state, planes) > 0) {
   n_planes = THCudaTensor_size(state, planes, 0);
  }
  long n_spheres = 0;
  if(THCudaTensor_nDimension(state, spheres) > 0) {
    n_spheres = THCudaTensor_size(state, spheres, 0);
  }

  long length = img_height * img_width;
  icgCudaRenderRender<<<GET_BLOCKS(length), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      THCudaTensor_data(state, img), length, img_height, img_width, 
      cam_x, cam_y, cam_z, step_x, step_y,
      n_planes, THCudaTensor_data(state, planes),
      n_spheres, THCudaTensor_data(state, spheres),
      bg_val);
  
  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg icgrender_IcgCudaRender__ [] = {
  {"IcgCudaRender_render", icgrender_IcgCudaRender_render},
  {NULL, NULL}
};

void icgrender_IcgCudaRender_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");

#if LUA_VERSION_NUM == 501
  luaL_register(L, NULL, icgrender_IcgCudaRender__); 
#else 
  luaL_setfuncs(L, icgrender_IcgCudaRender__, 0);
#endif
  // luaT_registeratname(L, icgrender_IcgRender__, "icgrender");

  // lua_pop(L, 1);
}

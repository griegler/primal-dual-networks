-- Copyright (C) 2016 Gernot Riegler
-- Institute for Computer Graphics and Vision (ICG)
-- Graz University of Technology (TU GRAZ)

-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are met:
-- 1. Redistributions of source code must retain the above copyright
--    notice, this list of conditions and the following disclaimer.
-- 2. Redistributions in binary form must reproduce the above copyright
--    notice, this list of conditions and the following disclaimer in the
--    documentation and/or other materials provided with the distribution.
-- 3. All advertising materials mentioning features or use of this software
--    must display the following acknowledgement:
--    This product includes software developed by the ICG, TU GRAZ.
-- 4. Neither the name of the ICG, TU GRAZ nor the
--    names of its contributors may be used to endorse or promote products
--    derived from this software without specific prior written permission.

-- THIS SOFTWARE IS PROVIDED ''AS IS'' AND ANY
-- EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
-- WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
-- DISCLAIMED. IN NO EVENT SHALL THE PROVIDER BE LIABLE FOR ANY
-- DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
-- (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
-- LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
-- ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
-- (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
-- SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

require "cutorch"

icgrender = {}


local _lib = require('./build/libicgrender')
for k, v in pairs(_lib) do icgrender[k] = v end


function icgrender.create_sphere(p0, r)
  return torch.Tensor{p0[1], p0[2], p0[3], r}
end 

function icgrender.create_plane(p0, v1, v2)
  local p = torch.cat(p0, p0 + v1, 2):cat(p0 + v1 + v2, 2):cat(p0 + v2, 2)
  return p:t()
end 

function icgrender.create_cube(p0, height, width, depth)
  local vx = torch.Tensor{width, 0, 0}
  local vy = torch.Tensor{0, height, 0}
  local vz = torch.Tensor{0, 0, depth}

  local c = torch.Tensor(6, 4, 3)
  c[1] = icgrender.create_plane(p0, vy, vx) -- front
  c[2] = icgrender.create_plane(p0 + vz, vy, vx) -- back
  c[3] = icgrender.create_plane(p0, vy, vz) -- left
  c[4] = icgrender.create_plane(p0 + vx, vy, vz) -- right
  c[5] = icgrender.create_plane(p0, vz, vx) -- bottom
  c[6] = icgrender.create_plane(p0 + vy, vz, vx) -- top

  return c
end

function icgrender.translate_plane(p, t)
  if p:dim() == 2 then 
    for pt_idx = 1, p:size(1) do 
      p[pt_idx]:add(t)
    end
  elseif p:dim() == 3 then 
    for pl_idx = 1, p:size(1) do
      for pt_idx = 1, p:size(2) do 
        p[{pl_idx, pt_idx}]:add(t)
      end
    end 
  end 

  return p
end 

function icgrender.center_plane(p)
  if p:dim() == 2 then 
    return p:mean(1):squeeze()
  elseif p:dim() == 3 then 
    return p:mean(1):mean(2):squeeze()
  end
end 

function icgrender.rotate_plane(p, phi_x, phi_y, phi_z, centered_rot)
  local centered_rot = centered_rot or true

  local sin, cos = math.sin, math.cos
  local Rx = torch.Tensor{{1, 0, 0}, {0, cos(phi_x), -sin(phi_x)}, {0, sin(phi_x), cos(phi_x)}}
  local Ry = torch.Tensor{{cos(phi_y), 0, sin(phi_y)}, {0, 1, 0}, {-sin(phi_y), 0, cos(phi_y)}}
  local Rz = torch.Tensor{{cos(phi_z), -sin(phi_z), 0}, {sin(phi_z), cos(phi_z), 0}, {0, 0, 1}}
  local R = Rx * Ry * Rz 
  
  local t = 0
  if centered_rot then t = -icgrender.center_plane(p) end

  icgrender.translate_plane(p, t)
  if p:dim() == 2 then 
    for pt_idx = 1, p:size(1) do 
      p[pt_idx] = p[pt_idx]:reshape(1, 3) * R
    end
  elseif p:dim() == 3 then 
    for pl_idx = 1, p:size(1) do
      for pt_idx = 1, p:size(2) do 
        p[{pl_idx, pt_idx}] = p[{pl_idx, pt_idx}]:reshape(1, 3) * R
      end
    end
  end
  icgrender.translate_plane(p, -t)

  return p
end 


include('IcgRender.lua')
include('IcgLuaRender.lua')
include('IcgCudaRender.lua')

return icgrender

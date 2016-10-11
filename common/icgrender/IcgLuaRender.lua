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

local IcgLuaRender, Parent = torch.class('icgrender.IcgLuaRender', 'icgrender.IcgRender')

function IcgLuaRender:__init(cam_pt, res, step, bg_val)
  Parent.__init(self, cam_pt, res, step, bg_val)
end 

function IcgLuaRender:inter_line_rect(r0, rd, p, debug)
  local debug = debug or false

  -- compute plane of rectangle
  local x1, x2, x3 = p[{1, 1}], p[{2, 1}], p[{3, 1}]
  local y1, y2, y3 = p[{1, 2}], p[{2, 2}], p[{3, 2}]
  local z1, z2, z3 = p[{1, 3}], p[{2, 3}], p[{3, 3}]
  local A = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2) 
  local B = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2) 
  local C = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
  local D = -x1 * (y2*z3 - y3*z2) - x2 * (y3*z1 - y1*z3) - x3 * (y1*z2 - y2*z1)

  -- compute point of intersection (line/plane)
  local x0, y0, z0 = r0[1], r0[2], r0[3]
  local xd, yd, zd = rd[1], rd[2], rd[3]
  local t = -(A * x0 + B * y0 + C * z0 + D) / (A * xd + B * yd + C * zd)
  local pi = r0 + rd * t

  if debug then 
    print(A, B, C, D)
    print('t=' .. t)
    print('pi=', pi)
  end

  if t >= 0 then 
    local v1 = p[1]:dot(p[2] - p[1])
    local v2 = pi:dot(p[2] - p[1])
    local v3 = p[2]:dot(p[2] - p[1])
    local v4 = p[1]:dot(p[4] - p[1])
    local v5 = pi:dot(p[4] - p[1])
    local v6 = p[4]:dot(p[4] - p[1])
    local within = (v1 <= v2 and v2 <= v3) and (v4 <= v5 and v5 <= v6)
    
    if within then 
      -- print('pi in rect')
      return pi 
    else 
      -- print('pi NOT in rect')
      return nil 
    end
  else 
    -- print('no intersection')
    return nil
  end
end

function IcgLuaRender:inter_line_sphere(r0, rd, sphere)
  local rd = rd / rd:norm()
  local c = sphere[{{1, 3}}]
  local r = sphere[4]

  local r0_m_c = r0 - c
  local radicand = rd:dot(r0_m_c)^2 - r0_m_c:dot(r0_m_c) + r^2

  if radicand < 0 then return nil end 

  local t1 = -rd:dot(r0_m_c) + math.sqrt(radicand)
  local pi1 = r0 + rd * t1

  local t2 = -rd:dot(r0_m_c) - math.sqrt(radicand)
  local pi2 = r0 + rd * t2

  local pi1_m_r0 = pi1 - r0 
  local pi2_m_r0 = pi2 - r0 
  if pi1_m_r0:dot(pi1_m_r0) < pi2_m_r0:dot(pi2_m_r0) then
    return pi1 
  else 
    return pi2 
  end
end 


function IcgLuaRender:render(planes, spheres)
  local rd = torch.Tensor{0, 0, 1}

  local r0_z = self.cam_z
  local r0_y = self.cam_y
  for row = self.img_height, 1, -1 do
    local r0_x = self.cam_x
    for col = 1, self.img_width do 
      local r0 = torch.Tensor{r0_x, r0_y, r0_z}

      local min_d = self.bg_val 
      for pl_idx = 1, planes:size(1) do
        local pi = self:inter_line_rect(r0, rd, planes[pl_idx])
        if pi and pi[3] < min_d then 
          min_d = pi[3]
        end
      end

      for sp_idx = 1, spheres:size(1) do 
        local pi = self:inter_line_sphere(r0, rd, spheres[sp_idx])
        if pi and pi[3] < min_d then 
          min_d = pi[3] 
        end
      end

      if min_d < 0 then min_d = 0 end
      self.img[{row, col}] = min_d

      r0_x = r0_x + self.step_x
    end 
    r0_y = r0_y + self.step_y
  end

  return self.img
end

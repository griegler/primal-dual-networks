#!/usr/bin/env th

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

local host = io.popen('uname -snr'):read('*line')
local data_root
if string.find(host, 'rvlab-.gr') then 
  data_root = '/home/gernotriegler/mount/nas5/riegler/dsrnet/'
  package.path = "/home/gernotriegler/Documents/projects/rv_utils/torch/?/init.lua;" .. package.path
else 
  error('unknown host: '..host)
end

require('hdf5')
require('paths')
require('icgrender')

-- setting
local cmd = torch.CmdLine()
cmd:text()
cmd:text('options:')
cmd:option('-n_files', 20, '')
cmd:option('-n_imgs_p_file', 2048, '')
cmd:option('-data_root', data_root, '')
cmd:option('-prefix', 'train', '')
cmd:option('-setting', '', '')
cmd:text()

local opt = cmd:parse(arg)


-- advanced settings
local rnd = torch.uniform
opt.img_height = 512
opt.img_width = 512
opt.vinvert = true
opt.cubes_min = 16
opt.cubes_max = 32
opt.cube_dim = function() return rnd(0.01, 0.6), rnd(0.01, 0.6), rnd(0.01, 0.6) end
opt.cube_t = function() return torch.Tensor{rnd(0.0, 1.0), rnd(0.0, 1.0), rnd(0.2, 1.0)} end
opt.spheres_min = 0
opt.spheres_max = 4
opt.sphere = function() return icgrender.create_sphere({rnd(0.1, 0.8), rnd(0.1, 0.8), rnd(0.1, 1.5)}, rnd(0.1, 0.3)) end
if opt.setting == 'middlebury' then 
  opt.vmin = 70
  opt.vmax = 218 
  opt.vinvert = true
elseif opt.setting == 'middlebury2' then 
  opt.vmin = 24
  opt.vmax = 224 
  opt.vinvert = true
elseif opt.setting == 'tofmark' then 
  opt.img_height = 610
  opt.img_width = 810
  opt.vmin = 400
  opt.vmax = 1000 
  opt.vinvert = false
else 
  error('unknown setting')
end

opt.res = {opt.img_height, opt.img_width}
opt.step = {1 / opt.img_height, 1 / opt.img_width}
opt.cam_pt = {0, 0, 0}
opt.bg_val = 1



-- init renderer and allocate cuda memory
local renderer = icgrender.IcgCudaRender(opt.cam_pt, opt.res, opt.step, opt.bg_val)
local imgs = torch.CudaTensor(opt.n_imgs_p_file, opt.res[1], opt.res[2])

-- render images
for file_idx = 1, opt.n_files do
  print()
  print('create data for file '..file_idx..' of '..opt.n_files..' with '..opt.n_imgs_p_file..' renderings')

  timer = torch.Timer()
  for img_idx = 1, opt.n_imgs_p_file do

    -- cubes
    local n_cubes = math.floor(rnd(opt.cubes_min, opt.cubes_max) + 0.5)
    local cubes = {}
    for n = 1, n_cubes do
      local height, width, depth = opt.cube_dim()
      local cube = icgrender.create_cube(torch.Tensor{0, 0, 0}, height, width, depth)

      local phi_x, phi_y, phi_z = rnd(-math.pi, math.pi), rnd(-math.pi, math.pi), rnd(-math.pi, math.pi)
      cube = icgrender.rotate_plane(cube, phi_x, phi_y, phi_z)
      

      -- find centroid
      local centroid = torch.Tensor{0, 0, 0}
      for plane_idx = 1, 6 do
        for pt_idx = 1, 4 do 
          centroid:add(cube[{plane_idx, pt_idx}])
        end
      end 
      centroid:div(24)

      -- translate cube to destination
      local cube_t = opt.cube_t()
      local t = cube_t - centroid
      cube = icgrender.translate_plane(cube, t)

      table.insert(cubes, cube)
    end 
    local planes = torch.cat(cubes[1], cubes[2], 1)
    for n = 3, n_cubes do planes = torch.cat(planes, cubes[n], 1) end

    -- spheres
    local n_spheres = math.floor(rnd(opt.spheres_min, opt.spheres_max) + 0.5)
    local spheres = torch.Tensor(n_spheres, 4)
    for n = 1, n_spheres do
      spheres[n] = opt.sphere()
    end

    -- render  
    renderer:setImg(imgs[img_idx])
    local img_gpu = renderer:render(planes:cuda(), spheres:cuda())
  end
  print('rendering took '..timer:time().real..'[s]')
  
  -- scale according to setting
  if opt.vinvert then
    imgs = imgs:mul(-(opt.vmax - opt.vmin)):add(opt.vmax)
  else 
    imgs = imgs:mul(opt.vmax - opt.vmin):add(opt.vmin)
  end
  imgs = imgs:round()
  print(string.format('min/max %f/%f', imgs:min(), imgs:max()))

  -- write h5 file
  local h5_path = paths.concat(opt.data_root, opt.setting, string.format(opt.prefix..'_%04d.h5', file_idx))
  paths.mkdir(paths.dirname(h5_path))
  print('write file '..file_idx..' of '..opt.n_files.. ' to '..h5_path)

  local h5_file = hdf5.open(h5_path, 'w')
  local h5_options = hdf5.DataSetOptions()
  h5_options:setChunked(1000, 1, opt.res[1], opt.res[2])
  h5_options:setDeflate()
  h5_file:write('/ta_depth', imgs:float():view(opt.n_imgs_p_file, 1, opt.res[1], opt.res[2]), h5_options)
  h5_file:close()
end
  




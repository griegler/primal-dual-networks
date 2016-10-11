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

local model = {}


function model.load(opt)
  local script_path = paths.dirname(debug.getinfo(1, 'S').source:sub(2))
  local srcnn10 = dofile(paths.concat(script_path, 'srcnn10.lua'))
  opt = srcnn10.load(opt)

  if opt.init_net_weights_path and paths.filep(opt.init_net_weights_path) then
    print('init feature net weights from '..opt.init_net_weights_path)
    feature_net = torch.load(opt.init_net_weights_path)
  else
    error('invalid feature net weights path')
  end

  local atgvl2 = dofile(paths.concat(script_path, 'atgvl2.lua'))

  opt = atgvl2.load(opt)
  local atgvl2_net = opt.net 

  opt.net = nn.Sequential()
  opt.net:add(feature_net)
  opt.net:add(atgvl2_net)

  opt.model_name = 'srcnn10_atgvl2'  

  opt.target_data_net = icgnn.IcgNarrow(10)   

  return opt
end


return model

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
  local backend 
  if opt.cudnn then 
    backend = cudnn 
  else 
    backend = nn 
  end

  local function conv(ci, co, fs, log_path)
    if log_path and log_path ~= '' and false then 
      local probe_params = {}
      probe_params['log_path'] = log_path
      probe_params['log_train'] = true
      return icgnn.IcgProbe(backend.SpatialConvolution(ci, co, fs, fs), probe_params)
    else
      return backend.SpatialConvolution(ci, co, fs, fs)
    end
  end

  local input = nn.Identity()()

  opt.narrow = 10
  
  local lr = icgnn.IcgResample('bilinear', opt.scale, opt.scale)(input)
  feature = backend.SpatialBatchNormalization(64)( backend.ReLU(true)( (conv( 1, 64, 3, 'layer1.h5')(lr)) ))
  feature = backend.SpatialBatchNormalization(64)( backend.ReLU(true)( conv(64, 64, 3, 'layer2.h5')(feature)) )
  feature = backend.SpatialBatchNormalization(64)( backend.ReLU(true)( conv(64, 64, 3, 'layer3.h5')(feature)) )
  feature = backend.SpatialBatchNormalization(64)( backend.ReLU(true)( conv(64, 64, 3, 'layer4.h5')(feature)) )
  feature = backend.SpatialBatchNormalization(64)( backend.ReLU(true)( conv(64, 64, 3, 'layer5.h5')(feature)) )
  feature = backend.SpatialBatchNormalization(64)( backend.ReLU(true)( conv(64, 64, 3, 'layer6.h5')(feature)) )
  feature = backend.SpatialBatchNormalization(64)( backend.ReLU(true)( conv(64, 64, 3, 'layer7.h5')(feature)) )
  feature = backend.SpatialBatchNormalization(64)( backend.ReLU(true)( conv(64, 64, 3, 'layer8.h5')(feature)) )
  feature = backend.SpatialBatchNormalization(64)( backend.ReLU(true)( conv(64, 64, 3, 'layer9.h5')(feature)) )

  local output_res = conv(64, 1, 3, 'layer10_1.h5')(feature)
  local output_t = nn.CAddTable()({icgnn.IcgNarrow(opt.narrow)(lr), output_res})
  local output_e = conv(64, 2, 3, 'layer10_2.h5')(feature)
  opt.net = nn.gModule({input}, {output_t, output_e})


  opt.criterion = nn.ParallelCriterion()
  opt.criterion:add(nn.MSECriterion())
  opt.criterion:add(nn.MSECriterion())

  opt.model_name = 'srcnn10_bn'
  opt.optimizer = optim['sgd']
  opt.learningRate = 1e-3
  opt.clip_gradient = 0.1
  opt.train_mask_border = 2

  return opt
end

return model

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

require('settings')


-- settings
opt.ex_db_path = paths.concat(opt.data_root, 'experiments.db')
opt.ex_method = 'srcnn10'
opt.ex_dataset = 'mb2_x4'
opt.experiment_name = opt.ex_dataset..'_'..opt.ex_method
opt.cudnn = true
opt.train_gpu = true
opt.test_gpu = true
opt.scale = 4
opt.input_datasets = {'ta_depth'}
opt.target_datasets = {'ta_depth'}
opt.net_model = 'models/srcnn10.lua'
opt.batch_size = 128
opt.momentum = 0.9
opt.epoch = 30
opt.learningRate_step = 30
opt.learningRate_factor = 0.1
opt.save_interval = 1

opt.ex_root = paths.concat(opt.data_root, opt.experiment_name)
print('ex_root: '..opt.ex_root)
paths.mkdir(opt.ex_root)
opt.out_prefix = paths.concat(opt.ex_root, 'out')
opt.image_names = {{'Cones'}, {'Teddy'}, {'Tsukuba'}, {'Venus'}}

opt.train_input_h5_paths = icgnn.listH5(opt.data_root, 'middlebury2/ph32_pw32_train')
opt.test_input_h5_paths = icgnn.listH5(opt.data_root, 'middlebury2/scale'..opt.scale..'_img')

-- log training
local log_path = paths.concat(opt.ex_root, 'run.log')
print('log to '..log_path)
local cmd = torch.CmdLine()
cmd:log(log_path, params)
for k, v in pairs(opt) do print(string.format('opt.%s = %s', k, v)) end

-- load model
local model = require(opt.net_model)
opt = model.load(opt)

-- input net
local input_data_net = nn.Sequential()
  :add(icgnn.IcgResample('bicubic', 1/opt.scale, 1/opt.scale, true))
input_data_net:cuda()
opt.input_data_preprocess_fcn = function(x) return input_data_net:forward(x) end

-- target net
local target = nn.Identity()()   
local target_e = icgnn.IcgNabla()(target)
local target_data_net = nn.gModule({target}, {icgnn.IcgNarrow(opt.narrow)(target), icgnn.IcgNarrow(opt.narrow)(target_e)})
target_data_net:cuda()
opt.target_data_preprocess_fcn = function(x) return target_data_net:forward(x) end

-- set up metrics
opt.metrics = {}
opt.metrics['rmse'] = function (input, target) return math.sqrt(torch.pow(input - target, 2):mean()) end
opt.metrics['sad'] = function (input, target) return torch.abs(input - target):mean() end


-- test if weights already exist, otherwise train network
local last_net_path = paths.concat(opt.ex_root, 'weights_epoch'..opt.epoch..'.t7')
if paths.filep(last_net_path) and true then
  opt.net = torch.load(last_net_path)

  print('test network')
  opt.ex_run = opt.epoch + 1
  icgnn.test(opt)
else
  print('test epoch init')
  opt.ex_run = 0
  icgnn.test(opt)   

  for epoch = 1, opt.epoch do
    print('epoch #' .. epoch .. ' of ' .. opt.epoch .. ' | lr = ' .. opt.learningRate)
    local tim = torch.Timer()
    icgnn.train(opt)
    print('epoch #'..epoch..' took '..tim:time().real..'[s]')

    print('test after train epoch '..epoch)
    opt.ex_run = epoch
    icgnn.test(opt)   

    -- save latest network
    if epoch % opt.save_interval == 0 or (epoch == opt.epoch) then
      local net_path = paths.concat(opt.ex_root, 'weights_epoch'..epoch..'.t7')
      torch.save(net_path, opt.net:clearState())
      print('saved model to: ' .. net_path)
    end  

    -- adjust lr
    if epoch % opt.learningRate_step == 0 then 
      opt.learningRate = opt.learningRate * opt.learningRate_factor 
    end
  end 
end




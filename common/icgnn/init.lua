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

require('torch')
require('nn')
require('paths')
require('./build/libicgnn')

icgnn = {}

include('IcgAddition.lua')
include('IcgCAddTable.lua')
include('IcgAddConstants.lua')
include('IcgThreshold.lua')
include('IcgThreshold2.lua')
include('IcgThreshold3.lua')

include('IcgExpMul.lua')

include('IcgNarrow.lua')

include('IcgNabla.lua')
include('IcgNablaT.lua')
include('IcgGeneralizedNabla.lua')
include('IcgGeneralizedNablaT.lua')

include('IcgL2Norm.lua')

include('IcgNoise.lua')

include('IcgResample.lua')
include('IcgMask.lua')


function icgnn.listFiles(root_dir, prefix, extension, debug) 
  local extension = extension or ''
  if #prefix > 0 then 
    root_dir = paths.concat(root_dir, paths.dirname(prefix))
    prefix = paths.basename(prefix)..'.*.'..extension
  else 
    root_dir = paths.concat(root_dir)
    prefix = ' *.'..extension 
  end 

  if debug then 
    print('listFiles - root_dir: '.. root_dir)
    print('listFiles - prefix: '.. prefix)
  end

  local h5_paths = {}
  for p in paths.files(root_dir, prefix) do 
    p = paths.concat(root_dir, p)
    table.insert(h5_paths, p) 
    if debug then print('listFiles - insert p: '.. p) end
  end 

  return h5_paths
end

function icgnn.listH5(root_dir, prefix, debug) 
  return icgnn.listFiles(root_dir, prefix, 'h5', debug)
end

function icgnn.h5Size(h5_paths, dataset_name)
  local hdf5 = require('hdf5')

  local size 
  local n_samples = 0
  for _, h5_path in ipairs(h5_paths) do
    local h5f = hdf5.open(h5_path, 'r')
    local h5_size = h5f:read(dataset_name):dataspaceSize()
    h5f:close()

    if size then
      size[1] = size[1] + h5_size[1] 
    else 
      size = h5_size 
    end 
  end

  return size
end

function icgnn.maskBorder(d, border)
  local function maskBorder(d, border)
    if border < 1 then 
      error('border has to be >= 1')
    end

    local hidx, widx
    if d:nDimension() == 2 then 
      hidx, widx = 1, 2
    elseif d:nDimension() == 3 then 
      hidx, widx = 2, 3
    elseif d:nDimension() == 4 then 
      hidx, widx = 3, 4
    else 
      error(string.format('invalid dim=%d for narrowBorder', d:nDim()))
    end

    local h, w = d:size(hidx), d:size(widx)

    d:narrow(hidx, 1, border):fill(0)
    d:narrow(hidx, h-border+1, border):fill(0)
    d:narrow(widx, 1, border):fill(0)
    d:narrow(widx, w-border+1, border):fill(0)

    return d
  end

  if torch.type(d) == 'table' then 
    local res = {}
    for idx = 1, #d do
      table.insert(res, maskBorder(d[idx], border))
    end
  else 
    return maskBorder(d, border)
  end
end 

function icgnn.narrowBorderAs(d, as)
  local function narrow(d, as)
    if d:nDimension() ~= as:nDimension() then 
      error('dimensions of d and as do not agreee')
    end

    local hidx, widx
    if d:nDimension() == 2 then 
      hidx, widx = 1, 2
    elseif d:nDimension() == 3 then 
      hidx, widx = 2, 3
    elseif d:nDimension() == 4 then 
      hidx, widx = 3, 4
    else 
      error(string.format('invalid dim=%d for narrowBorder', d:nDim()))
    end

    local h1, w1 = d:size(hidx), d:size(widx)
    local h2, w2 = as:size(hidx), as:size(widx)
    
    local h_border = h1 - h2 
    local h_border_l = math.floor(h_border / 2)
    local w_border = w1 - w2 
    local w_border_l = math.floor(w_border / 2)

    return d:narrow(hidx, 1+h_border_l, h1-h_border):narrow(widx, 1+w_border_l, w1-w_border)
  end

  if torch.type(d) == 'table' then 
    local res = {}
    for idx = 1, #d do 
      if torch.type(as) == 'table' then 
        table.insert(res, narrow(d[idx], as[idx]))
      else 
        table.insert(res, narrow(d[idx], as))
      end 
    end 
    return res
  else 
    if torch.type(as) == 'table' then
      return narrow(d, as[1])
    else
      return narrow(d, as)
    end
  end
end


function icgnn.init_weights_xavier_caffe(fan_in, fan_out)
  return math.sqrt(1 / fan_in)
end

function icgnn.init_weights_kaiming(fan_in, fan_out)
  return math.sqrt(4 / (fan_in + fan_out))
end

function icgnn.init_weights(net, method_name)
  local method = nil
  if method_name == 'xavier' then method = icgnn.init_weights_xavier_caffe
  elseif method_name == 'kaiming' then method = icgnn.init_weights_kaiming 
  else error('unknown init method: '..method_name) end

  local function init(m)
    if m then
      if torch.type(m) == 'nn.SpatialConvolution' or torch.type(m) == 'cudnn.SpatialConvolution' then
        m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
      elseif torch.type(m) == 'nn.SpatialConvolutionMM' then
        m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
      elseif torch.type(m) == 'nn.LateralConvolution' then
        m:reset(method(m.nInputPlane*1*1, m.nOutputPlane*1*1))
      elseif torch.type(m) == 'nn.VerticalConvolution' then
        m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW))
      elseif torch.type(m) == 'nn.HorizontalConvolution' then
        m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW))
      elseif torch.type(m) == 'nn.Linear' then
        m:reset(method(m.weight:size(2), m.weight:size(1)))
      elseif torch.type(m) == 'nn.TemporalConvolution' then
        m:reset(method(m.weight:size(2), m.weight:size(1)))    
      elseif torch.type(m) == 'icgnn.IcgAdjustLearningRate' then
        init(m.module)          
      elseif torch.type(m) == 'nn.gModule' then
        for node_idx, node in ipairs(m.forwardnodes) do -- do recursion 
          init(node.data.module)
        end
      end

      if m.bias then
        m.bias:zero()
      end
    end
  end

  for i, m in ipairs(net:listModules()) do
    local module_type = torch.type(m)
    init(m)
  end
end 


local function copy_LongStorage(sz)
  local new_sz = torch.LongStorage():resize(data_sz:size()):copy(data_sz)
  return new_sz
end 

local function table_to_LongStorage(sz)
  local new_sz = torch.LongStorage():resize(#sz)
  for idx, v in ipairs(sz) do
    new_sz[idx] = v
  end
  return new_sz
end


local function iterH5_(h5_paths, datasets, batch_size, gpu)
  local batch = {}

  local n_batches
  for idx = 1, #datasets do 
    local dataset_size = table_to_LongStorage(icgnn.h5Size(h5_paths, datasets[idx]))
    local n_samples = dataset_size[1]
    dataset_size[1] = batch_size
    
    if gpu then 
      batch[idx] = torch.CudaTensor(dataset_size)
    else
      batch[idx] = torch.FloatTensor(dataset_size)
    end 

    if batch_size <= n_samples then
      n_batches = math.floor(n_samples / batch_size)
    else
      error('number of samples is lower than batch_size')
    end
  end

  local data = {}
  local path_idx = 1 -- data comes from one h5 file, therefor one idx is sufficient
  local data_idx = 1 -- same here
  for batch_idx = 1, n_batches do 
    for n = 1, batch_size do 
      -- check if we need to load new data
      if data[1] == nil or data_idx > data[1]:size(1) then
        for dataset_idx, dataset in ipairs(datasets) do 
          local h5 = hdf5.open(h5_paths[path_idx], 'r')
          data[dataset_idx] = h5:read(dataset):all()
          h5:close()
        end
        path_idx = path_idx + 1
        data_idx = 1
      end

      -- copy data to batch
      for dataset_idx, dataset in ipairs(datasets) do 
        if batch_size == 1 then 
          -- print('  batch size == 1')
          local data_sz = data[dataset_idx]:size()
          local data_sz = torch.LongStorage():resize(data_sz:size()):copy(data_sz)
          data_sz[1] = 1
          batch[dataset_idx]:resize(data_sz)
        end
        batch[dataset_idx][n]:copy(data[dataset_idx][data_idx])
      end 
      data_idx = data_idx + 1
    end

    if #batch == 1 then
      coroutine.yield(batch[1])
    else 
      coroutine.yield(batch)
    end
  end
end 

function icgnn.iterH5(h5_paths, datasets, batch_size, gpu)
  return coroutine.wrap(function() iterH5_(h5_paths, datasets, batch_size, gpu) end)
end


function icgnn.train(opt)
  local net = opt['net'] or error('missing net param')
  local criterion = opt['criterion'] or error('missing criterion param')
  local input_h5_paths = opt['train_input_h5_paths'] or opt['input_h5_paths'] or error('missing input_h5_paths param')
  local target_h5_paths = opt['train_target_h5_paths'] or opt['target_h5_paths'] or input_h5_paths
  local input_datasets = opt['train_input_datasets'] or opt['input_datasets'] or error('missing input_datasets param')
  local target_datasets = opt['train_target_datasets'] or opt['target_datasets'] or input_datasets

  local input_data_preprocess_fcn = opt['train_input_data_preprocess_fcn'] or opt['input_data_preprocess_fcn'] or function(x) return x end
  local target_data_preprocess_fcn = opt['train_target_data_preprocess_fcn'] or opt['target_data_preprocess_fcn'] or function(x) return x end
  local optimizer = opt['optimizer'] or optim.sgd 
  local batch_size = opt['batch_size'] or 128
  local gpu = opt['train_gpu'] or opt['gpu'] or false
  local mask_border = opt['train_mask_border'] or opt['mask_border'] or 0
  local clip_gradient = opt['clip_gradient']
  
  local post_process_fcn = opt['post_process_fcn'] or nil

  local net_eval_fcn = opt['net_eval_fcn'] or nil

  local max_iter = opt['max_iter'] or -1

  -- prepare net and stuff for training on gpu
  net:training()
  
  if gpu then
    net:cuda()
    criterion:cuda()
    if opt.cudnn then
      cudnn.convert(net, cudnn)
    end
  else 
    net:float()
    criterion:float()
  end

  
  local parameters, gradParameters = net:getParameters()

  -- determine number of batches
  local n_samples = icgnn.h5Size(input_h5_paths, input_datasets[1])[1]
  for idx = 2, #input_datasets do 
    if n_samples ~= icgnn.h5Size(input_h5_paths, input_datasets[idx])[1] then 
      error(string.format('number of samples differ for dataset %s and %s', input_datasets[1], input_datasets[idx]))
    end
  end 
  for idx = 1, #target_datasets do 
    if n_samples ~= icgnn.h5Size(input_h5_paths, target_datasets[idx])[1] then 
      error(string.format('number of samples differ for dataset %s and %s', input_datasets[1], target_datasets[idx]))
    end
  end 

  local n_batches = math.floor(n_samples / batch_size)
  if max_iter > 0 then
    n_batches = max_iter 
  end

  -- get data iterator
  local input_iter = icgnn.iterH5(input_h5_paths, input_datasets, batch_size, gpu)
  local target_iter = icgnn.iterH5(target_h5_paths, target_datasets, batch_size, gpu)

  -- iterate over samples
  for batch_idx = 1, n_batches do 
    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end 
      gradParameters:zero()

      -- load the data
      local input_data = input_iter()
      local target_data = target_iter()

      local input = input_data_preprocess_fcn(input_data)
      local target = target_data_preprocess_fcn(target_data)

      -- evaluate network
      local f = nil
      if net_eval_fcn then
        f = net_eval_fcn(net, criterion, input, target)
      else
        local output = net:forward(input)

        -- post process net output and target
        if post_process_fcn then post_process_fcn(output, target) end

        f = criterion:forward(output, target)
        local df_do = criterion:backward(output, target)

        if mask_border > 0 then 
          icgnn.maskBorder(df_do, mask_border)
        end

        net:backward(input, df_do) 
      end

      if clip_gradient then
        if clip_gradient > 0 then 
          gradParameters:clamp(-clip_gradient, clip_gradient)
        elseif clip_gradient < 0 then
          local cg = -clip_gradient / opt.learningRate
          gradParameters:clamp(-cg, cg)
        end
      end 
      
      -- debug output
      if batch_idx < 65 or batch_idx % math.floor((n_batches / 200)) == 0 then 
        print(string.format('iter=%4d | loss=%9.6f ', batch_idx, f))
      end

      return f, gradParameters
    end 
    optimizer(feval, parameters, opt)
    xlua.progress(batch_idx, n_batches)

  end
end

function icgnn.test(opt)
  local net = opt['net'] or error('missing net param')
  local metrics = opt['metrics'] or error('missing metrics param')
  local input_h5_paths = opt['test_input_h5_paths'] or opt['input_h5_paths'] or error('missing input_h5_paths param')
  local target_h5_paths = opt['test_target_h5_paths'] or opt['target_h5_paths'] or input_h5_paths
  local mask_h5_paths = opt['test_mask_h5_paths'] or opt['mask_h5_paths'] or target_h5_paths
  local input_datasets = opt['test_input_datasets'] or opt['input_datasets'] or error('missing input_datasets param')
  local target_datasets = opt['test_target_datasets'] or opt['target_datasets'] or input_datasets
  local mask_datasets = opt['mask_datasets']
  local image_names = opt['image_names']

  local input_data_preprocess_fcn = opt['test_input_data_preprocess_fcn'] or opt['input_data_preprocess_fcn'] or function(x) return x end
  local target_data_preprocess_fcn = opt['test_target_data_preprocess_fcn'] or opt['target_data_preprocess_fcn'] or function(x) return x end
  local gpu = opt['test_gpu'] or opt['gpu'] or false
  local mask_border = opt['test_mask_border'] or opt['mask_border'] or 0

  local net_forward_fcn = opt['net_forward_fcn'] or nil
  local post_process_fcn = opt['post_process_fcn'] or nil
  local store_h5_fcn = opt['store_h5_fcn'] or nil

  local ex_run = opt['ex_run'] or 0
  local out_prefix = opt['out_prefix']
  local store_h5_out = (opt['store_h5_out'] == nil) or opt['store_h5_out'] -- default is true
  local sample_inc = opt['sample_inc'] or 1


  -- prepare net and stuff for training on gpu
  net:evaluate()
  
  if gpu then
    net:cuda()
  else 
    if cudnn then
      cudnn.convert(net, nn)
    end
    net:float()
  end

  
  -- determine number of batches
  local n_samples = icgnn.h5Size(input_h5_paths, input_datasets[1])[1]
  for idx = 2, #input_datasets do 
    if n_samples ~= icgnn.h5Size(input_h5_paths, input_datasets[idx])[1] then 
      error(string.format('number of samples differ for dataset %s and %s', input_datasets[1], input_datasets[idx]))
    end
  end 
  for idx = 1, #target_datasets do 
    if n_samples ~= icgnn.h5Size(input_h5_paths, target_datasets[idx])[1] then 
      error(string.format('number of samples differ for dataset %s and %s', input_datasets[1], target_datasets[idx]))
    end
  end 

  -- get data iterator
  local input_iter = icgnn.iterH5(input_h5_paths, input_datasets, 1, gpu)
  local target_iter = icgnn.iterH5(target_h5_paths, target_datasets, 1, gpu)
  local mask_iter
  if mask_datasets then
    mask_iter = icgnn.iterH5(mask_h5_paths, mask_datasets, 1, gpu)
  end

  -- iterate over samples
  local metric_sums = {}
  for batch_idx = 1, n_samples, sample_inc do 
    -- load the data
    local input_data = input_iter()
    local target_data = target_iter()
    local mask
    if mask_iter then
      mask = mask_iter()
      if torch.type(mask) ~= 'table' then mask = {mask} end
    end

    local input = input_data_preprocess_fcn(input_data)
    local target = target_data_preprocess_fcn(target_data)

    -- evaluate network
    local output = nil
    if net_forward_fcn then 
      output = net_forward_fcn(net, input)
    else
      output = net:forward(input)
    end

    -- convert output to table
    if torch.type(output) ~= 'table' then output = {output} end
    if torch.type(target) ~= 'table' then target = {target} end

    -- post process net output and target
    if post_process_fcn then post_process_fcn(output, target) end

    -- mask
    if mask_border > 0 then 
      icgnn.maskBorder(output, mask_border)
      icgnn.maskBorder(target, mask_border)
    end

    if mask then
      for oidx = 1, math.min(#output, #mask) do 
        for c = 1, output[oidx]:size(2) do 
          output[oidx][{{}, c, {}, {}}]:cmul(icgnn.narrowBorderAs(mask[oidx], output[oidx]))
        end
      end
      for tidx = 1, math.min(#target, #mask) do 
        for c = 1, target[tidx]:size(2) do 
          target[tidx][{{}, c, {}, {}}]:cmul(icgnn.narrowBorderAs(mask[tidx], target[tidx]))
        end
      end
    end 

    -- debug output
    if batch_idx == 1 then 
      for table_idx = 1, #target do
        local out_str = 'target '..table_idx..' has dim '
        for dim_idx = 1, target[table_idx]:nDimension() do 
          out_str = out_str..target[table_idx]:size(dim_idx)..'x'
        end
        print(out_str)
      end
      for table_idx = 1, #output do
        if torch.isTensor(output[table_idx]) then
          local out_str = 'output '..table_idx..' has dim '
          for dim_idx = 1, output[table_idx]:nDimension() do 
            out_str = out_str..output[table_idx]:size(dim_idx)..'x'
          end
          print(out_str)
        end
      end
    end 

    -- store outputs in h5
    if out_prefix and store_h5_out then
      if store_h5_fcn then
        store_h5_fcn(input, output, target, batch_idx, ex_run)
      else
        for table_idx = 1, math.max(#target, #output) do
          if torch.isTensor(output[table_idx]) then
            local image_name = 'b'..batch_idx..'_t'..table_idx
            if image_names and image_names[batch_idx] and image_names[batch_idx][table_idx] then
              image_name = image_names[batch_idx][table_idx]
            end 
            local h5_path = out_prefix..'_'..image_name..'_run'..ex_run..'.h5'
            local h5_file = hdf5.open(h5_path, 'w')
            if output[table_idx] then
              h5_file:write('/output', output[table_idx]:float())
            end
            if target[table_idx] then
              h5_file:write('/target', target[table_idx]:float())
            end
            h5_file:close()
          end
        end
      end
    end

    -- evaluatue metrics
    for table_idx = 1, math.min(#target, #output) do 
      if next(metrics) ~= nil and torch.isTensor(output[table_idx]) then
        local metric_str = '| b='..batch_idx..' | t='..table_idx..' | '
        for metric_name, metric in pairs(metrics) do 
          -- print(output[table_idx]:size())
          -- print(target[table_idx]:size())
          local metric_val = metric(output[table_idx], target[table_idx])
          if metric_sums[metric_name] then
            metric_sums[metric_name] = metric_sums[metric_name] + metric_val
          else 
            metric_sums[metric_name] = metric_val 
          end
          metric_str = metric_str..metric_name..'='..metric_val..' | '

          if opt.ex_db_path and opt.ex_method and opt.ex_dataset then
            local image_name = 'b'..batch_idx..'_t'..table_idx
            if image_names and image_names[batch_idx] and image_names[batch_idx][table_idx] then
              image_name = image_names[batch_idx][table_idx]
            end 
            local h5_path = out_prefix..'_'..image_name..'.h5'
            icgnn.logExperiment(opt.ex_db_path, opt.ex_method, opt.ex_dataset, image_name, metric_name, metric_val, h5_path, '', ex_run)
          end --if
        end  -- for metric_name, metric
        print(metric_str)
      end -- if torch.isTensor
    end
  end    
  
  local metric_means = {}
  local metric_str = '| mean | '
  for metric_name, metric_sum in pairs(metric_sums) do 
    local metric_mean = metric_sum / n_samples
    table.insert(metric_means, metric_mean)
    metric_str = metric_str..metric_name..'='..metric_mean..' | '
  end 
  if metric_str ~= '| mean | ' then print(metric_str) end

  return metric_means
end


function icgnn.saveNet(path, net)
  torch.save(path, net:clearState())
end 

function icgnn.saveNetWeights(path, net)
  local params, grad_params = net:getParameters()
  torch.save(path, params)
end 

function icgnn.cudnn2cunn(net)
  -------------------------------------------------------------------------------
  -- these functions are adapted from Michael Partheil
  -- https://groups.google.com/forum/#!topic/torch7/i8sJYlgQPeA
  -- the problem is that you can't call :float() on cudnn module, it won't convert
  local function replaceModules(net, orig_class_name, replacer)
    local nodes, container_nodes = net:findModules(orig_class_name)
    for i = 1, #nodes do
      for j = 1, #(container_nodes[i].modules) do
        if container_nodes[i].modules[j] == nodes[i] then
          local orig_mod = container_nodes[i].modules[j]
          print('replacing a cudnn module with nn equivalent...')
          print(orig_mod)
          container_nodes[i].modules[j] = replacer(orig_mod)
        end
      end
    end
  end
  local function cudnnNetToCpu(net)
    local net_cpu = net:clone():float()
    replaceModules(net_cpu, 'cudnn.SpatialConvolution', 
      function(orig_mod)
        local cpu_mod = nn.SpatialConvolution(orig_mod.nInputPlane, orig_mod.nOutputPlane,
            orig_mod.kW, orig_mod.kH, orig_mod.dW, orig_mod.dH, orig_mod.padW, orig_mod.padH)
        cpu_mod.weight:copy(orig_mod.weight)
        cpu_mod.bias:copy(orig_mod.bias)
        cpu_mod.gradWeight = nil -- sanitize for thinner checkpoint
        cpu_mod.gradBias = nil -- sanitize for thinner checkpoint
        return cpu_mod
      end)
    replaceModules(net_cpu, 'cudnn.SpatialMaxPooling', 
      function(orig_mod)
        local cpu_mod = nn.SpatialMaxPooling(orig_mod.kW, orig_mod.kH, orig_mod.dW, orig_mod.dH, 
                                             orig_mod.padW, orig_mod.padH)
        return cpu_mod
      end)
    replaceModules(net_cpu, 'cudnn.ReLU', function() return nn.ReLU() end)
    return net_cpu
  end
  
  return cudnnNetToCpu(net)
end

function icgnn.optimizeInferenceMemory(net)
-- a function to do memory optimizations by 
-- setting up double-buffering across the network.
-- this drastically reduces the memory needed.
-- based on https://github.com/soumith/dcgan.torch/blob/master/util.lua#L85-L101

  local finput, output, outputB
  net:apply(
    function(m)
      if torch.type(m):find('Convolution') then
        finput = finput or m.finput
        m.finput = finput
        output = output or m.output
        m.output = output
      elseif torch.type(m):find('ReLU') then
        m.inplace = true
      elseif torch.type(m):find('BatchNormalization') then
        outputB = outputB or m.output
        m.output = outputB
      end
  end)

  return net
end




local function createDb(path)
  if not paths.filep(path) then
    local sql = require('lsqlite3')
    local db = sql.open(path)

    db:exec[[
      CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        method TEXT NOT NULL,
        dataset TEXT NOT NULL,
        image TEXT NOT NULL,
        run INTEGER NOT NULL DEFAULT 1,
        metric TEXT NOT NULL,
        value REAL NOT NULL,
        file TEXT NOT NULL DEFAULT '',
        comment TEXT NOT NULL DEFAULT '',
        ts INTEGER DEFAULT CURRENT_TIMESTAMP NOT NULL,
        CONSTRAINT unique_entry UNIQUE (method, dataset, image, run, metric) ON CONFLICT REPLACE
      )
    ]]

    db:close()
  end
end

function icgnn.logExperiment(db_path, method, dataset, image, metric, value, file, comment, run)
  -- print('log experiment to '..db_path..' method='..method..' dataset='..dataset..' image='..image..' metric='..metric..' run='..run)
  createDb(db_path)

  local sql = require('lsqlite3')
  local db = sql.open(db_path)
  db:busy_timeout(5000)
  local err_code = db:exec("INSERT INTO experiments (method, dataset, image, run, metric, value, file, comment) " ..
          "VALUES('"..method.."', '"..dataset.."', '"..image.."', '"..run.."', '"..metric.."', '"..value.."', '"..file.."', '"..comment.."')")
  -- print(err_code)
  if err_code ~= 0 then
    print('WARNING: logExperiment failed with sqlite error code '..err_code)
  end
  local experiment_id = db:last_insert_rowid()

  db:close()

  return experiment_id
end 


return icgnn


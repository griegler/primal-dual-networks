local model = {}

function model.load(opt)
  local backend 
  if opt.cudnn then 
    backend = cudnn 
  else 
    backend = nn 
  end

  opt.narrow = 10

  opt.net = nn.Sequential()
    :add( nn.ParallelTable()
      :add( icgnn.IcgResample('bilinear', opt.scale, opt.scale) )
      :add( nn.Identity() )
    )
    :add( nn.ConcatTable()
      :add(
        nn.Sequential()
          :add( nn.JoinTable(1, 3) )
          :add( backend.SpatialConvolution( 2, 64, 3, 3) )
          :add( backend.ReLU(true) )
          :add( backend.SpatialConvolution(64, 64, 3, 3) )
          :add( backend.ReLU(true) )
          :add( backend.SpatialConvolution(64, 64, 3, 3) )
          :add( backend.ReLU(true) )
          :add( backend.SpatialConvolution(64, 64, 3, 3) )
          :add( backend.ReLU(true) )
          :add( backend.SpatialConvolution(64, 64, 3, 3) )
          :add( backend.ReLU(true) )
          :add( backend.SpatialConvolution(64, 64, 3, 3) )
          :add( backend.ReLU(true) )
          :add( backend.SpatialConvolution(64, 64, 3, 3) )
          :add( backend.ReLU(true) )
          :add( backend.SpatialConvolution(64, 64, 3, 3) )
          :add( backend.ReLU(true) )
          :add( backend.SpatialConvolution(64, 64, 3, 3) )
          :add( backend.ReLU(true) )
          :add( nn.ConcatTable()
              :add( backend.SpatialConvolution(64, 1, 3, 3) )
              :add( backend.SpatialConvolution(64, 24, 3, 3) )
          )
      )
      :add( nn.SelectTable(1) )
    )
    :add( nn.FlattenTable() )
    :add( nn.ConcatTable()
      :add( nn.Sequential()
        :add( nn.ConcatTable()
          :add( nn.SelectTable(1) )
          :add( nn.Sequential():add(nn.SelectTable(3)):add(icgnn.IcgNarrow(opt.narrow)) )
        )
        :add( nn.CAddTable() )
      )
      :add( nn.SelectTable(2) )
    )


  opt.criterion = nn.ParallelCriterion()
  opt.criterion:add(nn.MSECriterion())
  opt.criterion:add(nn.SmoothL1Criterion())

  opt.model_name = 'dc_de124_srcnn10'
  opt.optimizer = optim['sgd']
  opt.learningRate = 1e-3
  opt.clip_gradient = 0.1

  return opt
end

return model

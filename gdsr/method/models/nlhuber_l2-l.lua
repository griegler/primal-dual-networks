local model = {}

  
function model.load(opt)
  local params = opt.variational or {}
  local affinity = params.affinity or error('no affinity')
  local max_iter = params.max_iter or 20
  local lambda = params.lambda or 0.00005
  local eps = params.eps or 0.01
  local sigma = params.sigma or 1.0/16.0
  local tau = params.tau or 8.0 / 16.0
  local theta = params.theta or 1.0
  local sigma_d = params.sigma_d or 2
  local sigma_g = params.sigma_g or 0.01

  local ip = true
  

  local net = nn.Sequential()

  -- prepare W matrix and init ubar, u, p
  local input = nn.Identity()()
  local f = nn.SelectTable(1)(input)
  local g = nn.SelectTable(2)(input)

  local W = nn.Abs()(g)
  local W_mul = nn.Mul()
  W_mul.weight:fill(-1.0 / (255 * sigma_g))
  W = W_mul( W )
  W = cudnn.SpatialSoftMax()(W)
  W = nn.AddConstant(1e-12, ip)( W )

  local ubar = nn.Identity()(f)
  local u = nn.Identity()(f)
  local p = icgnn.IcgGeneralizedNabla(affinity)(f)

  net:add( nn.gModule({input}, {u, ubar, p, f, W}) )

  -- iterations
  for iter = 1, max_iter do
    local p_denom = 1 + sigma * eps
    local u_denom = 1.0 + tau * lambda

    local iter_input = nn.Identity()()
    local u = nn.SelectTable(1)(iter_input)
    local ubar = nn.SelectTable(2)(iter_input)
    local p = nn.SelectTable(3)(iter_input)
    local f = nn.SelectTable(4)(iter_input)
    local W = nn.SelectTable(5)(iter_input)

    -- dual update
    p = icgnn.IcgCAddTable({1 / p_denom, sigma / p_denom}, true)({p, icgnn.IcgGeneralizedNabla(affinity)(ubar)})
    -- projection
    local norm = nn.CDivTable()({nn.Abs()(p), W})
    norm = nn.Threshold(1, 1, true)(norm)
    p = nn.CDivTable()({p, norm})
   
    -- primal update
    local un = icgnn.IcgCAddTable({1.0 / u_denom, tau / u_denom, tau * lambda / u_denom}, true)({ 
      u, icgnn.IcgGeneralizedNablaT(true, affinity)(p), f
    })
    
    -- over relaxation
    if iter < max_iter then
      ubar = icgnn.IcgCAddTable({(1 + theta), -1.0}, true)({ un, u })
      net:add( nn.gModule({iter_input}, {un, ubar, p, f, W}) )
    else
      net:add( nn.gModule({iter_input}, {un}) )
    end
  end

  opt.net = net

  opt.criterion = nn.MSECriterion()

  opt.model_name = 'nlhuber_l2'
  opt.optimizer = optim['sgd']
  opt.learningRate = 1e-4
  opt.clip_gradient = 0.1

  return opt
end

return model

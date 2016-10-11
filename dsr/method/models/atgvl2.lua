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

local function atgvl2(max_iter, lambda, edge_w, tgv_alpha0, tgv_alpha1, tensor_beta, tensor_gamma)
  -- parameters 
  local max_iter = max_iter or 10
  local lambda = lambda or 1
  local edge_w = edge_w or 1
  local tgv_alpha0 = tgv_alpha0 or 17
  local tgv_alpha1 = tgv_alpha1 or 1.2
  local tensor_beta = tensor_beta or 10
  local tensor_gamma = tensor_gamma or 0.75

  local eta_p = 3
  local eta_q = 2

  local tau = 1;
  local sigma = 1 / tau;
  local timestep_lambda = 1

  local lambda_modules = {}
  local param_modules = {}

  -- split net output estimate/edge
  local input = nn.Identity()()
  local u_init = nn.SelectTable(1)(input)
  local edge = icgnn.IcgExpMul()
  edge.weight:fill(math.log(edge_w))
  param_modules['edge_mul'] = edge
  edge = edge(nn.SelectTable(2)(input))


  local tensor_abs = icgnn.IcgL2Norm()(edge)
  local n_normed = nn.CDivTable()({edge, nn.Replicate(2, 1, 3)( nn.Threshold(0, 1e-6)(tensor_abs) )}) 
  local nT_normed = icgnn.IcgVectorRotate()(n_normed)
  local wtensor = nn.Threshold(1e-8, 1e-8, true)( nn.Exp()( nn.MulConstant(-tensor_beta, true)( nn.Power(tensor_gamma)(tensor_abs) )) )
  local n_normed_split = nn.SplitTable(1, 3)(n_normed)
  local nT_normed_split = nn.SplitTable(1, 3)(nT_normed)
  local atensor = nn.CAddTable()({nn.CMulTable()({wtensor, nn.Power(2)(nn.SelectTable(1)(n_normed_split))}), nn.Power(2)(nn.SelectTable(1)(nT_normed_split))})
  local ctensor = nn.CAddTable()({nn.CMulTable()({wtensor, nn.SelectTable(1)(n_normed_split), nn.SelectTable(2)(n_normed_split)}), nn.CMulTable()({nn.SelectTable(1)(nT_normed_split), nn.SelectTable(2)(nT_normed_split)})})
  local btensor = nn.CAddTable()({nn.CMulTable()({wtensor, nn.Power(2)(nn.SelectTable(2)(n_normed_split))}), nn.Power(2)(nn.SelectTable(2)(nT_normed_split))})

  local atensor2 = nn.Power(2)(atensor)
  local btensor2 = nn.Power(2)(btensor)
  local ctensor2 = nn.Power(2)(ctensor)
  local eta_u = nn.Threshold(1e-8, 1e-8, false)( nn.MulConstant(tgv_alpha1^2, true)( nn.CAddTable()({atensor2, btensor2, nn.MulConstant(2, false)(ctensor2), nn.Power(2)(nn.CAddTable()({atensor, ctensor})), nn.Power(2)(nn.CAddTable()({btensor, ctensor}))}) ) )
  local eta_v1 = nn.AddConstant(4*tgv_alpha0^2, true)(icgnn.IcgAddition({tgv_alpha1^2, tgv_alpha1^2})({btensor2, ctensor2}))
  local eta_v2 = nn.AddConstant(4*tgv_alpha0^2, true)(icgnn.IcgAddition({tgv_alpha1^2, tgv_alpha1^2})({atensor2, ctensor2}))
  local eta_v = nn.JoinTable(1, 3)({eta_v1, eta_v2})

  -- iterations
  local d = nn.Identity()(u_init)
  local u = nn.Identity()(d)
  local u_ = nn.Identity()(u)
  local v_, p_x, p_y, q
  local u_x, u_y, du_tensor_x, du_tensor_y, projection_p, Tp_x, Tp_y, dq_tensor, div_p
  for iter = 1, max_iter do 
    local mu = 1 / math.sqrt(1 + 0.7 * tau * timestep_lambda)
    if sigma >= 1e3 then mu = 1 end 

    -- dual variables update
    local u_xy = nil
    if iter == 1 then 
      u_xy = nn.SplitTable(1, 3)( icgnn.IcgNabla()(u_) )
    else 
      u_xy = nn.SplitTable(1, 3)( icgnn.IcgAddition({1, -1})({icgnn.IcgNabla()(u_), v_}) )
    end 
    u_x = nn.SelectTable(1)(u_xy)
    u_y = nn.SelectTable(2)(u_xy)

    du_tensor_x = nn.CAddTable()({nn.CMulTable()({atensor, u_x}), nn.CMulTable()({ctensor, u_y})})
    du_tensor_y = nn.CAddTable()({nn.CMulTable()({ctensor, u_x}), nn.CMulTable()({btensor, u_y})})
      
    if iter == 1 then 
      p_x = nn.MulConstant(tgv_alpha1 * sigma / eta_p, true)(du_tensor_x)
      p_y = nn.MulConstant(tgv_alpha1 * sigma / eta_p, true)(du_tensor_y)
    else 
      p_x = nn.CAddTable()({p_x, nn.MulConstant(tgv_alpha1 * sigma / eta_p, true)(du_tensor_x)})
      p_y = nn.CAddTable()({p_y, nn.MulConstant(tgv_alpha1 * sigma / eta_p, true)(du_tensor_y)})
    end

    projection_p = nn.Threshold(1, 1, true)(nn.Sqrt()(nn.CAddTable()({nn.Power(2)(p_x), nn.Power(2)(p_y)})))
    p_x = nn.CDivTable()({p_x, projection_p})
    p_y = nn.CDivTable()({p_y, projection_p})

    if iter > 1 and iter < max_iter then
      local grad_v = icgnn.IcgNabla()(v_)
      if iter == 2 then 
        q = nn.MulConstant(tgv_alpha0 * sigma / eta_q, true)(grad_v)
      else 
        q = nn.CAddTable()({q, nn.MulConstant(tgv_alpha0 * sigma / eta_q, true)(grad_v)})
      end

      local projection_q = nn.Threshold(1, 1, true)(icgnn.IcgL2Norm()(q))
      q = nn.CDivTable()({q, nn.Replicate(4)(projection_q)})
    end

    -- primal variables update
    u_ = nn.Identity()(u)
    if iter >= 2 and iter < max_iter then
      v_ = nn.Identity()(v)
    end

    Tp_x = nn.CAddTable()({nn.CMulTable()({atensor, p_x}), nn.CMulTable()({ctensor, p_y})})
    Tp_y = nn.CAddTable()({nn.CMulTable()({ctensor, p_x}), nn.CMulTable()({btensor, p_y})})
    dq_tensor = nn.JoinTable(1, 3)({Tp_x, Tp_y})
    
    div_p = icgnn.IcgNablaT(true)( dq_tensor )


    local lambda_numerator
    if #lambda_modules == 0 then 
      lambda_numerator = icgnn.IcgExpMul()
    else
      lambda_numerator = icgnn.IcgExpMul()
    end
    table.insert(lambda_modules, lambda_numerator)
    param_modules['lambda_n_'..iter] = lambda_numerator
    local lambda_denominator = icgnn.IcgExpMul()
    table.insert(lambda_modules, lambda_denominator)
    param_modules['lambda_d_'..iter] = lambda_denominator
    local numerator = nn.CAddTable()({ u_, nn.CDivTable()({ nn.CAddTable()({nn.MulConstant(tau * tgv_alpha1, true)(div_p), nn.MulConstant(tau, false)(lambda_numerator(d))}), eta_u }) })
    local denominator = nn.AddConstant(1, true)(nn.MulConstant(tau, true)(lambda_denominator(nn.Power(-1)(eta_u))))
    u = nn.CDivTable()({numerator, denominator})

    if iter < max_iter then
      if iter == 1 then 
        v = nn.CDivTable()({ nn.MulConstant(tau * tgv_alpha1, true)(dq_tensor), eta_v })
      else
        local div_q = icgnn.IcgNablaT(true)(q)
        v = nn.CAddTable()({ v_, nn.CDivTable()({ icgnn.IcgAddition({tau * tgv_alpha1, tau * tgv_alpha0})({dq_tensor, div_q}), eta_v }) })
      end
    end

    -- over relaxation 
    u_ = icgnn.IcgAddition({1 + mu, -mu})({u, u_})
    if iter < max_iter then
      if iter == 1 then 
        v_ = nn.MulConstant(1 + mu, false)(v)
      else
        v_ = icgnn.IcgAddition({1 + mu, -mu})({v, v_})
      end
    end

    sigma = sigma / mu
    tau = tau * mu
  end

  -- set lambda parameter on modules
  for _, lambda_module in ipairs(lambda_modules) do
    lambda_module.weight:fill(math.log(lambda))
  end

  local net = nn.gModule({input}, {u_})
  return net, param_modules
end



function model.load(opt)
  local backend 
  if opt.cudnn then 
    backend = cudnn 
  else 
    backend = nn 
  end

  opt.model_name = 'atgvl2'
  opt.criterion = nn.MSECriterion()
  opt.optimizer = optim['sgd']
  opt.learningRate = 1e-4
  opt.clip_gradient = 0.1
  opt.batch_size = 32

  local atgvl2_net, param_modules = atgvl2(unpack(opt.atgvl2_params))
  opt.net = atgvl2_net
  opt.param_modules = param_modules

  return opt
end


return model


-- print('-----------------------------------------')
-- for k, mod in pairs(param_modules) do
--   print(k)
--   print(mod.weight[1], math.exp(mod.weight[1]))
-- end

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

local IcgAddition, Parent = torch.class('icgnn.IcgAddition', 'nn.Module')

function IcgAddition:__init(alphas)
  Parent.__init(self)
  assert(type(alphas) == 'table', 'alphas is not a table')
  self.alphas = alphas

  self.gradInput = {}
end 

function IcgAddition:updateOutput(input)
  assert(type(input) == 'table' or torch.isTensor(input), 'input is not a table, or tensor; type = ' .. type(input) .. ')')

  if type(input) == 'table' then 
    assert(#input == #self.alphas, '#inputs(' .. #input .. ') ~= #alphas(' .. #self.alphas .. ')')
    self.output:resizeAs(input[1])

    self.output:fill(0)
    for idx = 1, #input do
      self.output:add(self.alphas[idx], input[idx])
    end
  elseif torch.isTensor(input) then 
    assert(#self.alphas == 1, '#inputs(1) ~= #alphas(' .. #self.alphas .. ')')
    self.output:resizeAs(input)
    self.output:mul(input, self.alphas[1])
  end 

  return self.output
end 

function IcgAddition:updateGradInput(input, gradOutput)
  for idx = 1, #self.alphas do 
    self.gradInput[idx] = self.gradInput[idx] or input[idx].new()
    self.gradInput[idx]:resizeAs(input[idx])
    self.gradInput[idx]:mul(gradOutput, self.alphas[idx])
  end

  return self.gradInput
end


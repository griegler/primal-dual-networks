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

local IcgAddConstants, parent = torch.class('icgnn.IcgAddConstants', 'nn.Module')

function IcgAddConstants:__init(scalars, ip, dimension, nInputDims)
  parent.__init(self)
  assert(type(scalars) == 'table', 'input is not table!')
  self.scalars = scalars
  
  -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end

   self.dimension = dimension or 1
   self.nInputDims = nInputDims
end


function IcgAddConstants:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input:dim() + dimension + 1
   elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end


function IcgAddConstants:updateOutput(input)
  local dim = self:_getPositiveDimension(input)
  assert(input:size(dim) == #self.scalars, 'number of channels does not match number of scalars')

  if self.inplace then
    self.output:set(input)
  else
    self.output:resizeAs(input)
    self.output:copy(input)
  end

  for idx, scalar in ipairs(self.scalars) do
    self.output:narrow(dim, idx, 1):add(scalar)
  end 

  return self.output
end 

function IcgAddConstants:updateGradInput(input, gradOutput)
  local dim = self:_getPositiveDimension(input)
  assert(input:size(dim) == #self.scalars, 'number of channels does not match number of scalars')

  if self.inplace then
    self.gradInput:set(gradOutput)
    -- restore previous input value
    for idx, scalar in ipairs(self.scalars) do
      input:narrow(dim, idx, 1):add(-scalar)
    end
  else
    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)
  end

  return self.gradInput
end

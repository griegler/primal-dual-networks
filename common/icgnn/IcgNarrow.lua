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

local IcgNarrow, Parent = torch.class('icgnn.IcgNarrow', 'nn.Module')

function IcgNarrow:__init(border)
  Parent.__init(self)

  self.border = border or 0
end 

function IcgNarrow:updateOutput(input)
  local hidx, widx
  if input:nDimension() == 2 then 
    hidx, widx = 1, 2
  elseif input:nDimension() == 3 then 
    hidx, widx = 2, 3
  elseif input:nDimension() == 4 then 
    hidx, widx = 3, 4
  else 
    error(string.format('invalid dim=%d for IcgNarrow', d:nDim()))
  end

  local h, w = input:size(hidx), input:size(widx)
  if 2 * self.border >= h or 2 * self.border >= w then 
    error(string.format('self.border(%d) is to large for input %dx%d', self.border, h, w))
  end

  self.output = input:narrow(hidx, 1+self.border, h-2*self.border):narrow(widx, 1+self.border, w-2*self.border)

  return self.output
end 

function IcgNarrow:updateGradInput(input, gradOutput)
  local hidx, widx
  if input:nDimension() == 2 then 
    hidx, widx = 1, 2
  elseif input:nDimension() == 3 then 
    hidx, widx = 2, 3
  elseif input:nDimension() == 4 then 
    hidx, widx = 3, 4
  else 
    error(string.format('invalid dim=%d for IcgNarrow', d:nDim()))
  end

  local h, w = input:size(hidx), input:size(widx)

  self.gradInput:resizeAs(input):fill(0)
  local gi = self.gradInput:narrow(hidx, 1+self.border, h-2*self.border)
  gi = gi:narrow(widx, 1+self.border, w-2*self.border)
  gi:copy(gradOutput)

  return self.gradInput
end


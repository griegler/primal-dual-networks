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

local IcgGeneralizedNablaT, Parent = torch.class('icgnn.IcgGeneralizedNablaT', 'nn.Module')

function IcgGeneralizedNablaT:__init(neg, directions)
  Parent.__init(self)

  self.neg = neg or false
  self.directions = directions
end 

function IcgGeneralizedNablaT:updateOutput(input)
  assert(self.directions:dim() == 2, 'directions must be a 2 x N tensor')
  assert(self.directions:size(1) == 2, 'directions must be a 2 x N tensor')

  return input.icgnn.IcgGeneralizedNablaT_updateOutput(self, input)
end 

function IcgGeneralizedNablaT:updateGradInput(input, gradOutput)
  return input.icgnn.IcgGeneralizedNablaT_updateGradInput(self, input, gradOutput)
end


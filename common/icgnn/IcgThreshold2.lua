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

local IcgThreshold2, parent = torch.class('icgnn.IcgThreshold2', 'nn.Module')


-- out = input2 <= threshold ? val : input1
-- input1 ... input to threshold
-- input2 ... selector

function IcgThreshold2:__init(threshold, val)
  parent.__init(self)

  self.threshold = threshold or 1e-6
  self.val = val or 0

  self.gradInput = {}
end
 


function IcgThreshold2:updateOutput(input)
  if torch.type(input) ~= 'table' or #input ~= 2 then
    error('input has to be a table with 2 entries (type='..torch.type(input)..')')
  end

  input[1].icgnn.IcgThreshold2_updateOutput(self, input[1], input[2])
  return self.output 
end

function IcgThreshold2:updateGradInput(input, gradOutput) 
  if torch.type(input) ~= 'table' or #input ~= 2 then
    error('input has to be a table with 2 entries (type='..torch.type(input)..')')
  end 
  
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[2] = self.gradInput[2] or input[2].new()

  input[1].icgnn.IcgThreshold2_updateGradInput(self, input[1], input[2], self.gradInput[1], gradOutput)
  self.gradInput[2]:resizeAs(input[2]):fill(0)

  return self.gradInput
end


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

local IcgThreshold3, parent = torch.class('icgnn.IcgThreshold3', 'nn.Module')


-- out = input3 <= threshold ? input2 : input1
-- input1 ... value if input3 is larger threshold
-- input2 .. value if input3 is smaller threshold
-- input3 ... selector

function IcgThreshold3:__init(threshold)
  parent.__init(self)

  self.threshold = threshold or 1e-6

  self.gradInput = {}
end
 


function IcgThreshold3:updateOutput(input)
  if torch.type(input) ~= 'table' or #input ~= 3 then
    error('input has to be a table with 3 entries (type='..torch.type(input)..')')
  end

  input[1].icgnn.IcgThreshold3_updateOutput(self, input[1], input[2], input[3])
  return self.output 
end

function IcgThreshold3:updateGradInput(input, gradOutput) 
  if torch.type(input) ~= 'table' or #input ~= 3 then
    error('input has to be a table with 3 entries (type='..torch.type(input)..')')
  end 
  
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[2] = self.gradInput[2] or input[2].new()
  self.gradInput[3] = self.gradInput[3] or input[3].new()

  input[1].icgnn.IcgThreshold3_updateGradInput(self, input[1], input[2], input[3], self.gradInput[1], self.gradInput[2], gradOutput)
  self.gradInput[3]:resizeAs(input[1]):fill(0)

  return self.gradInput
end


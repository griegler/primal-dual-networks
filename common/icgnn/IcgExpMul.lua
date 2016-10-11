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

local IcgExpMul, parent = torch.class('icgnn.IcgExpMul', 'nn.Module')

function IcgExpMul:__init()
   parent.__init(self)
  
   self.weight = torch.Tensor(1)
   self.gradWeight = torch.Tensor(1)
   
   self:reset()
end
 
function IcgExpMul:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end

   self.weight[1] = torch.uniform(-stdv, stdv);
end

function IcgExpMul:updateOutput(input)
   self.output:resizeAs(input):copy(input);
   self.output:mul( math.exp(self.weight[1]) );
   return self.output 
end

function IcgExpMul:updateGradInput(input, gradOutput) 
   self.gradInput:resizeAs(input):zero()
   self.gradInput:add( math.exp(self.weight[1]), gradOutput )
   return self.gradInput
end

function IcgExpMul:accGradParameters(input, gradOutput, scale) 
   scale = scale or 1
   self.gradWeight[1] = self.gradWeight[1] + scale * math.exp(self.weight[1]) * input:dot(gradOutput);
end

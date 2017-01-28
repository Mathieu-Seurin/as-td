require 'nn'

function loadCrit(nOutput)

   local criterion = nn.ParallelCriterion()
   for crit=1,nOutput do
      criterion:add(nn.CrossEntropyCriterion())
   end
   return criterion
   
end

function createModel(nInput, numLayer)
   assert(nInput,"Need sizeInput (for criterion)")
   
   local inputFrameSize = 128
   local outputFrameSize = 128
   --fixed, number of note possible

   local numLayer = numLayer or 7
   local nOutput = nInput-numLayer

   local kw = 2
   local dw = 1

   local model = nn.Sequential()

   for lay=1,numLayer do
      model:add(nn.TemporalConvolution(inputFrameSize,outputFrameSize,kw,dw))
   end

   return model, loadCrit(nOutput)

end


-- inputFrameSize = 5

-- outputFrameSize = 5
-- kw = 2
-- dw = 1

-- a = torch.Tensor({{1,2,3,4,5},{1,2,3,4,5},{5,4,3,2,1},{5,4,3,2,1}})
-- b = torch.Tensor({{1,2,3,4,5},{1,2,3,4,5}})
-- c = torch.Tensor({{5,4,3,2,1},{5,4,3,2,1}})

-- tp = nn.TemporalConvolution(inputFrameSize,outputFrameSize,kw,dw)

-- output = tp:forward(a)
-- print(output)

-- output = tp:forward(b)
-- print(output)

-- output = tp:forward(c)
-- print(output)


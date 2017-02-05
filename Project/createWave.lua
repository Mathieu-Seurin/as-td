require 'nn'

function loadCrit(nOutput)

   local criterion = nn.ParallelCriterion()
   for crit=1,nOutput do
      criterion:add(nn.CrossEntropyCriterion())
   end
   return criterion
   
end

function createModel(nInput, numLayer, filterSz, filterStride)
   assert(nInput,"Need sizeInput (for criterion)")
   
   local inputFrameSize = 128
   local outputFrameSize = 128
   --fixed, number of note possible

   local numLayer = numLayer or 7

   local kw = filterSz
   local dw = filterStride

   local model = nn.Sequential()

   for lay=1,numLayer do
      if lay==1 then
         model:add(nn.TemporalConvolution(inputFrameSize,outputFrameSize,kw,dw))
         model:add(nn.ReLU())
      else
         model:add(nn.TemporalConvolution(inputFrameSize,outputFrameSize,2,1))
         model:add(nn.ReLU())

      end
   end

   seqX = torch.zeros(1,nInput,inputFrameSize)
   res = model:forward(seqX)

   nOutput = res:size(2)

   print("Time window :",nInput-nOutput)
   
   return model, loadCrit(nOutput), nOutput

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


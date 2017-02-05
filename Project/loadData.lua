matio = require 'matio'

MODELDIR = 'Model/'

function loadDummy()
   return matio.load('dummy.mat','mat')
end

function loadData(name)
   return matio.load(name,'mat')
end

function splitDataBatch(data,nInput,nOutput,numLayer,batchSize)

   numEx = data:size(1)
   numFeat = data:size(2)
   numBatch = numEx-nInput

   numNotIncluded = nInput-nOutput
   
   batches={}
   for i=1,numBatch-batchSize+1,batchSize do
      indexLow = i
      indexHigh= i+nInput-1
         
      batches[#batches+1] = {}
      batches[#batches]['x'] = torch.zeros(batchSize,nInput,numFeat)
      batches[#batches]['y'] = torch.zeros(batchSize,nOutput)

      for sz=0,batchSize-1 do
         batches[#batches]['x'][{sz+1,{},{}}] = data[{{indexLow+sz,indexHigh+sz},{}}]

         yTemp = data[{{indexLow+numNotIncluded+1+sz,indexHigh+1+sz},{}}]
         y = {}
         for i=1,yTemp:size(1) do
            y[#y+1] = torch.range(1,yTemp[i]:nElement())[torch.eq(yTemp[i],1)][1]
         end
         batches[#batches]['y'][{sz+1,{}}] = torch.Tensor(y)
         --print('y',batches[#batches]['y'])
      end

      batches[#batches]['x']:cuda()
      
   end

   return batches

end

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then
      io.close(f)
      return true
   else return false
   end
end

--TEST
--====

-- a = torch.Tensor({{1},{2},{3},{4},{5}})
-- splitDataNonDilated(a,2,1)

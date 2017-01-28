require 'nn'
require 'createWave'
require 'loadData'
require 'cunn'

local numFeatures = 128

local batchSize = 15
local learning_rate = 1e-2
local maxEpoch = 3

local numLayer = 50
local nOutput = 40

local nInput = numLayer+nOutput

--data = loadDummy()
data = loadData('paco.mat')

local numEx = data:size(1)
local numBatch = (numEx-nInput)/batchSize

batches=splitDataBatch(data,nInput,numLayer,batchSize)

-- ========= LOADING OR CREATING NETWORK
-- =====================================
if file_exists(MODELDIR..'WaveNet.model'..numLayer) then
   print("Network alread exist, loading ...")
   net = torch.load(MODELDIR..'WaveNet.model'..numLayer)
   net:cuda()
   crit = loadCrit(nOutput)

else
   print("Network doesn't exist, creating ...")
   net, crit = createModel(nInput,numLayer)
end

-- ========= TRAINING NETWORK =========
-- =====================================
print("Number of iteration per epoch :", math.floor(numBatch))

parameters,gradParameters = model:getParameters()

local feval = function(x)
   if x ~= parameters then
      parameters:copy(x)
   end
   
   gradParameters:zero()
   local f=0

   for i=1,numBatch do
      local id = torch.random(1,numBatch)
      local x, y = batches[id]['x'], batches[id]['y']

      local yhat = net:forward(x)

      yhat = yhat:reshape(nOutput,batchSize,numFeatures)
      y = y:reshape(nOutput,batchSize)
      
      local loss = crit:forward(yhat,y)
      f = f+loss
      print(loss)
      
      local delta = crit:backward(yhat,y):reshape(batchSize,nOutput,numFeatures)
      
      net:zeroGradParameters()
      net:backward(x,delta)
      net:updateParameters(learning_rate)

      return f,gradParameters

   end
end


for epoch = 1,maxEpoch do

      gradParameters:div(#inputs)
      f = f/#inputs

   end
   
   globalLoss = optimMethod(feval, parameters, optimState)

   print("globalLoss",globalLoss)
   print("LOSS ITERATION N°",epoch,' : ',globalLoss/numBatch/batchSize)

end

-- ====== SAVING AND TESTING NETWORK ===
-- =====================================
torch.save(MODELDIR..'WaveNet.model'..numLayer, net)


numTest = 1000
accuracy = 0

-- Not a real test since data can be taken from train
for ex = 1,numTest do
   local id = torch.random(1,numBatch)
   local x, y = batches[id]['x'], batches[id]['y']

   local yhat = net:forward(x)

   for i = 1,y:size(1) do

      local maxProba, indexMax = torch.max(yhat[i],1)
      -- print("yhat",yhat[i])
      -- print("indexMax",indexMax)
      -- print("y",y[i])
      -- io.read()
      if indexMax[1] == y[i] then
         accuracy = accuracy +1
      else
         print("id",indexMax[1])
         print("y",y[i])
      end
   end
end

print("accuracy",accuracy/(numTest*nOutput))


numTest = 1000
accuracy = 0

-- Random
for ex = 1,numTest do

   randChoice = torch.rand(nOutput)
   yhat = torch.zeros(nOutput,128)

   for i=1,nOutput do
      if randChoice[i] < 0.142858 then
         yhat[{i,46}] = 1
      else
         yhat[{i,41}] = 1
      end
   end
   
   for i = 1,nOutput do

      local maxProba, indexMax = torch.max(yhat[i],1)
      -- print("yhat",yhat[i])
      -- print("indexMax",indexMax)
      -- print("y",y[i])
      -- io.read()
      if indexMax[1] == y[i] then
         accuracy = accuracy +1
      end
   end
end

print("Rand accuracy",accuracy/(numTest*nOutput))

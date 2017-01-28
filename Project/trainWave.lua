require 'nn'
require 'createWave'
require 'loadData'
require 'cunn'

local numFeatures = 128

local batchSize = 1
local learning_rate = 1e-2
local maxEpoch = 10

local numLayer = 20
local nOutput = 2

local nInput = numLayer+nOutput

data = loadDummy()
--data = loadData('paco.mat')

local numEx = data:size(1)
local numBatch = (numEx-nInput)/batchSize

batches=splitDataBatch(data,nInput,numLayer,batchSize)

-- ========= LOADING OR CREATING NETWORK
-- =====================================
if file_exists(MODELDIR..'WaveNet.model'..numLayer) then
   print("Network alread exist, loading ...")
   net = torch.load(MODELDIR..'WaveNet.model'..numLayer)
   crit = loadCrit(nOutput)

else
   print("Network doesn't exist, creating ...")
   net, crit = createModel(nInput,numLayer)
end

-- ========= TRAINING NETWORK =========
-- =====================================
print("Number of iteration per epoch :", math.floor(numBatch))

for epoch = 1,maxEpoch do

   local globalLoss = 0
   
   for i=1,numBatch do
      local id = torch.random(1,numBatch)
      local x, y = batches[id]['x'], batches[id]['y']

      local yhat = net:forward(x)

      if i%10==0 then
         local maxProba, indexMax = torch.max(yhat[{1,{},{}}],2)
         print("id : ",indexMax:reshape(1,nOutput))
         print("y", y:reshape(1,nOutput))
      end
      
      yhat = yhat:reshape(nOutput,batchSize,numFeatures)
      y = y:reshape(nOutput,batchSize)

      local loss = crit:forward(yhat,y)

      print(loss)
      globalLoss = globalLoss + loss
      local delta = crit:backward(yhat,y):reshape(batchSize,nOutput,numFeatures)

      -- if i>100 then
      --    print("delta",delta[{1,{},{48,53}}])
      --    io.read()
      -- end

      net:zeroGradParameters()
      net:backward(x,delta)
      net:updateParameters(learning_rate)

   end

   print("globalLoss",globalLoss)
   print("LOSS ITERATION N°",epoch,' : ',globalLoss/numBatch/batchSize)

end

-- ====== SAVING AND TESTING NETWORK ===
-- =====================================
torch.save(MODELDIR..'WaveNet.model'..numLayer, net)

print("Testing Model :\n")
numTest = math.floor(1000/nOutput)
print("Number of test", numTest)

accuracy = 0

-- Not a real test since data is taken from train
for ex = 1,numTest do
   local id = torch.random(1,numBatch)
   local x, y = batches[id]['x'], batches[id]['y']

   local yhat = net:forward(x)[{1,{},{}}]

   local maxProba, indexMax = torch.max(yhat,2)

   for i = 1,nOutput do

      if indexMax[{i,1}] == y[{1,i}] then
         accuracy = accuracy +1
      end
   end
end

print("accuracy",accuracy/(numTest*nOutput))



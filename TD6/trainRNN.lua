require 'nn'
require 'nngraph'
require 'creationRNN'
require 'load_exemple'
require 'io'
require 'model_utils'

vocab = torch.load('vocab.t7')

local batchSize = 1
local M = 35
local T = 15
local N = 150
--local learning_rate = 0.01 -- Learning rate at the beginning
local learning_rate = 0.0005 -- Learning rate at the end
local maxEpoch = 10

--nngraph.setDebug(true) 


-- ========= LOADING OR CREATING NETWORK
-- =====================================
if file_exists('lovecraftRNN.model'..N) then
   print("Network alread exist, loading ...")
   net = torch.load('lovecraftRNN.model'..N)
   netPred = torch.load('lovecraftRNN.pred'..N)
   crit = loadCritRNN(T)

else
   print("Network doesn't exist, creating ...")
   net, crit, netPred = create_RNN(M,N,T)
   --graph.dot(net.fg, 'RNN', 'wholeRNN_FG')
   --graph.dot(net.bg, 'RNN', 'wholeRNN_BG')

end

data = splitBatch(batchSize,T)
cardBatches = #data.x_batches
numIte = 4000

print("Number of batches", cardBatches)
print("Number of iteration",numIte)
local ht = nil



-- ========= TRAINING NETWORK =========
-- =====================================
for epoch = 1,maxEpoch do

   local globalLoss = 0
   
   for i=1,numIte do
      local id = torch.random(1,cardBatches)
      local x, y = reformatBatch(data.x_batches[id],M,N,batchSize,T)
      local yhat = net:forward(x)

      -- REFERENCE : Random
      -- local yRand = generateRandomY(N,M,batchSize)
      -- local randLoss = crit:forward(yRand,y)
      local loss = crit:forward(yhat,y)

      
      -- print("randLoss",randLoss)
      -- print("loss",loss)

      globalLoss = globalLoss + loss
      local delta = crit:backward(yhat,y)

      net:zeroGradParameters()
      net:backward(x,delta)
      net:updateParameters(learning_rate)

      -- if i%100==0 then
      --    print("yhat1",yhat[T-1])
      --    print("y",y[T-1])
      --    print("yhat2",yhat[5])
      --    print("y",y[5])

      --    io.read()

      -- end
   end

   print("LOSS ITERATION N°",epoch,' : ',globalLoss/numIte/batchSize)

   --==== Checking if parameters are the same for the whole network and the predictionNet
   -- introspect_parameters(net)
   -- introspect_parameters(netPred)
   -- io.read()

end

-- ====== SAVING AND TESTING NETWORK ===
-- =====================================
torch.save('lovecraftRNN.model'..N, net)
torch.save('lovecraftRNN.pred'..N, netPred)

numTest = 1000
accuracy = 0

-- Not a real test since data can be taken from train
-- Right now, aroud 40% of good prediction
-- Random is 0.028%
for ex = 1,numTest do
   local id = torch.random(1,cardBatches)
   local x, y = reformatBatch(data.x_batches[id],M,N,batchSize,T)

   local yhat = net:forward(x)

   for i = 1,y:size(1) do
   
      local maxProba, indexMax = torch.max(yhat[i],2)
      -- print("yhat",yhat[i])
      -- print("indexMax",indexMax)
      -- print("y",y[i])
      -- io.read()
      if indexMax[1][1] == y[i][1] then
         accuracy = accuracy +1    
      end
   end
end


print("accuracy",accuracy/(numTest*(T-1)))

require 'nn'
require 'nngraph'
require 'creationRNN'
require 'load_exemple'
require 'io'
require 'model_utils'

vocab = torch.load('vocab.t7')

local batchSize = 1
local M = 35
local N = 3
local learning_rate = 0.01
local maxEpoch = 10

--nngraph.setDebug(true) 

if file_exists('lovecraftRNN.model'..N) then
   print("Network alread exist, loading ...")
   net = torch.load('lovecraftRNN.model'..N)
   netPred = ('lovecraftRNN.pred'..N)
   crit = loadCritRNN(N)

else
   print("Network doesn't exist, creating ...")
   net, crit, netPred = create_RNN(M,N)
   graph.dot(net.fg, 'RNN', 'wholeRNN_FG')
   graph.dot(net.bg, 'RNN', 'wholeRNN_BG')

end

data = splitBatch(batchSize,N)
cardBatches = #data.x_batches

print("Number of batches", cardBatches)

local ht = nil

for epoch = 1,maxEpoch do

   local globalLoss = 0
   local ht 

   
   for i=1,cardBatches do

      local x, y = reformatBatch(data.x_batches[i],M,N,batchSize,ht)
      local yhat = net:forward(x)

      -- REFERENCE : Random
      -- local yRand = generateRandomY(N,M,batchSize)
      -- local randLoss = crit:forward(yRand,y)


      local loss = crit:forward(yhat,y)

      -- print("randLoss",randLoss)
      -- print("loss",loss)
      print("x",x)
      print("y",y)
      
      globalLoss = globalLoss + loss
      local delta = crit:backward(yhat,y)

      net:zeroGradParameters()
      net:backward(x,delta)
      net:updateParameters(learning_rate)

      if i%100==0 then
         print("yhat1",yhat[1])
         print("y",y[1])
         print("yhat2",yhat[2])
         print("y",y[2])

         io.read()

      end


   end

   print("LOSS ITERATION N°",epoch,' : ',globalLoss/cardBatches)

   --==== Checking if parameters are the same for the whole network and the predictionNet
   -- introspect_parameters(net)
   -- introspect_parameters(netPred)
   -- io.read()

end



torch.save('lovecraftRNN.model'..N, net)
torch.save('lovecraftRNN.pred'..N, netPred)
torch.save('lovecraftRNN.ht'..N, ht)



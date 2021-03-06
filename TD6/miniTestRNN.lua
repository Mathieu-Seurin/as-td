require 'nn'
require 'nngraph'
require 'creationRNN'
require 'load_exemple'
require 'io'
require 'model_utils'

vocab = torch.load('vocab.t7')

local batchSize = 1
local M = 3
local N = 3
local learning_rate = 0.1
local maxEpoch = 10000

nngraph.setDebug(true) 


print("Network doesn't exist, creating ...")
net, crit, netPred = create_RNN(M,N)
graph.dot(net.fg, 'miniRNN', 'miniRNN')

local letter_batches = {}
local ytrad = {}
local init = torch.ones(1,N)
local lol = {}



if N == 2 then
   
   local a = torch.Tensor({{1,0}})
   local b = torch.Tensor({{0,1}})
   letter_batches = {{init,b,a},{init,a,a},{init,b,a}, {init,a,b}}
   ytrad = {{{1}},{{1}},{{1}},{{2}}}
   lol = {'ba', 'aa', 'ba', 'ab'}

else

   local a = torch.Tensor({{1,0,0}})
   local b = torch.Tensor({{0,1,0}})
   local c = torch.Tensor({{0,0,1}})

   letter_batches = {{init,b,a,b},{init,a,b,b},{init,a,a,a}, {init,b,b,a}}
   ytrad = {{{1,2}},{{2,2}},{{1,1}},{{2,1}}}
   lol = {'bab','abb','aaa', 'bba'}
end

local cardBatches = #letter_batches

print("Number of batches", cardBatches)

for epoch = 1,maxEpoch do

   local globalLoss = 0
   
   for i=1,cardBatches do

      local x, y = letter_batches[i], torch.Tensor(ytrad[i]):t()
      local yhat = net:forward(x)
      local loss = crit:forward(yhat,y)
      local delta = crit:backward(yhat,y)
      globalLoss = globalLoss + loss

      print("x",lol[i],i)
      print("yhat",yhat[2])
      print("y",y[2])
      print("loss",loss)
      print("delta",delta[2])
      io.read()

      net:zeroGradParameters()
      net:backward(x,delta)
      net:updateParameters(learning_rate)
      

      -- if epoch>5000 then
      --    print("x",x[2])
      --    print("yhat1",yhat)
      --    print("y",y[1])
      --    print("loss",loss)
      --    io.read()
      -- end

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



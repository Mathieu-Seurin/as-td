require 'nn'
require 'dataset'
require 'io'

local function sample(out)
   out = torch.exp(out)
   return torch.multinomial(out,1)
end


local function train(model,X,Y)

   local loss = 0
   local numEpoch = 100
   local sizeBatch = 10
   local cardExemple = X:size(1)
   local cardFeatures = X:size(2)
   local learningRate = 1e-2
   
   for epoch=1,numEpoch do
      for ex=1,cardExemple/sizeBatch do

         local x = torch.Tensor(sizeBatch, cardFeatures)
         local y = torch.Tensor(sizeBatch)

         for i=1,sizeBatch do
            randEx = torch.random(1,cardExemple)
            x[{i,{}}] = X[randEx]
            y[i] = Y[randEx]
         end
         
         local yhat = model:forward(x)
         local yhat = sample(yhat)

         local delta = torch.zeros(sizeBatch,cardFeatures)

         for i=1,yhat:size(1) do
            if yhat[i][1]~=y[i] then
               delta[{i,yhat[i][1]}] = 1
            end
         end

         model:zeroGradParameters()
         model:backward(x,delta)

         model:updateParameters(learningRate)
         
         loss = loss + delta:sum()
         
      end

      print("Loss : ",loss/sizeBatch)
      loss = 0

   end
end

N = 100
N = N*4
local X,y = createDataset(N)

local dim = 2
local sizeOut = 2

local model = nn.Sequential()
model:add(nn.Linear(dim,sizeOut))
model:add(nn.LogSoftMax())


train(model, X,y)

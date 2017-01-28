-- Policy gradient (Reinforcement)
-- ~ Deux gradients : Un pour modifier les paramètres theta, un autre pour transformer les probas d'echantillonnage.

require 'dpnn'
require 'nn'
require 'dataset'
require 'io'

require 'gnuplot'

local function sample(out)
   out = torch.exp(out)
   return torch.multinomial(out,1)
end

local function train(model,X,Y)

   local sumLoss = 0
   local loss = 0

   local numEpoch = 100
   local sizeBatch = 1
   local cardExemple = X:size(1)
   local cardFeatures = X:size(2)
   local learningRate = 1e-2

   c = nn.MSECriterion()
   
   for epoch=1,numEpoch do

      local prec = 0

      for ex=1,cardExemple/sizeBatch do

         local x = torch.Tensor(sizeBatch, cardFeatures)
         local y = torch.Tensor(sizeBatch,2)

         for i=1,sizeBatch do
            randEx = torch.random(1,cardExemple)
            x[{i,{}}] = X[randEx]
            y[i] = Y[randEx]
         end
         
         model:zeroGradParameters()


         local yhat = model:forward(x)
         local loss = c:forward(yhat,y)
         loss = torch.Tensor({loss})
         local delta = c:backward(yhat,y)

         -- print("delta",delta)
         -- print("yhat",yhat)
         -- print("y",y)
         -- io.read()
         
         model:reinforce(-loss)
         model:backward(x,delta)

         model:updateParameters(learningRate)
         
         sumLoss = sumLoss + loss
         
      end

      if epoch%5==0 then
         print("loss : ",sumLoss/sizeBatch)
      end
      sumLoss = 0
   end

   return model

end
   

N = 100
N = N*4
M = 100

local X,y = createDatasetVect(N)
local XTest,yTest = createDatasetVect(N)

print("yTest",yTest)
_,yTest = torch.max(yTest,2)
yTest = yTest:double()

local dim = 2
local N = 10

local model = nn.Sequential()
model:add(nn.Linear(dim,M))
model:add(nn.Sigmoid())

model:add(nn.ReinforceBernoulli())
model:add(nn.Linear(M,2))

yhat = model:forward(XTest)
_,yhat = torch.max(yhat,2)

yhat = yhat:double()
res = torch.eq(yhat,yTest)
res = res:double()
print("Accuracy Before Training",res:mean())

model = train(model,X,y)

yhat = model:forward(XTest)

_,yhat = torch.max(yhat,2)
yhat = yhat:double()
print("yhat",yhat)
print("yTest",yTest)
res = torch.eq(yhat,yTest)
res = res:double()
print("Accuracy",res:mean())



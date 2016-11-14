require 'nn'
require 'gnuplot'
require 'load_mnist'

require 'io'


------------ TOOLS : Accuracy/Split
-- =================================
local function accuracy(modelOutput, y)

   -- Matrix, return the whole accuracy
   modelOutput[torch.lt(modelOutput,0)] = -1 
   modelOutput[torch.gt(modelOutput,0)] = 1

   local resultMatrix = (torch.eq(modelOutput,y)):double()
   return resultMatrix:mean()

end

function SplitTrainTest(X,y,split)
   split = split or 0.8
   
   cardExemple = y:size(1)
   randomTensor = torch.randperm(cardExemple):long()
   sizeSplit = math.floor(split*cardExemple)
   
   posIndex = randomTensor[{{1,sizeSplit}}]
   negIndex = randomTensor[{{sizeSplit+1,randomTensor:size(1)}}]

   xTrain = X:index(1,posIndex)
   xTest = X:index(1,negIndex)
   yTrain = y:index(1,posIndex)
   yTest = y:index(1,negIndex)

   return xTrain,xTest,yTrain,yTest
end

function parseLab(str)

   if #str == 1 then
      numTable = {tonumber(str)}
      
   elseif str:sub(1,1) == '-' then
      return {-1}
      
   else
      numTable = {} 
      for i = 1, #opt.posLab do
         local c = str:sub(i,i)
         numTable[i] = tonumber(c)
      end
   end

   return numTable
end


-- CUSTOM CRITERION
--=================
local myHuberLoss,parent = torch.class('HuberLoss','nn.Criterion')

function HuberLoss:__init() -- constructeur
   -- parent.__init(self)
   self.gradInput = torch.Tensor()
   self.output = 0
end

function HuberLoss:forward(input, target) -- appel generique pour calculer le cout
   return self:updateOutput(input, target)
end

function HuberLoss:updateOutput(input, target)
   
   local preCalc = 1-input*target
   
   if preCalc <= 2 then
      self.output = torch.max(torch.Tensor({0, preCalc}))^2
   else
      self.output = -4*input*target
   end
   return self.output

end

function HuberLoss:backward(input, target) -- appel generique pour calculer le gradient du cout
   return self:updateGradInput(input, target)
end

function HuberLoss:updateGradInput(input, target) -- a completer

   local preCalc = 1-input*target

   if preCalc <= 2 then

      if torch.max(torch.Tensor({0, preCalc}))^2 == 0 then
         self.gradInput = torch.DoubleTensor({0})
      else
         self.gradInput = 2*(torch.pow(target,2))*input - 2*target
      end
   else
      self.gradInput = -4*target
   end

   return self.gradInput

end


local CustomLoss,parent = torch.class('SeurinLoss','nn.Criterion')

function SeurinLoss:__init() -- constructeur
   -- parent.__init(self)
   self.gradInput = torch.Tensor()
   self.output = 0
end

function SeurinLoss:forward(input, target) -- appel generique pour calculer le cout
   return self:updateOutput(input, target)
end

function SeurinLoss:updateOutput(input, target)
   
   return torch.abs(input - target)
end

function SeurinLoss:backward(input, target) -- appel generique pour calculer le gradient du cout
   return self:updateGradInput(input, target)
end

function SeurinLoss:updateGradInput(input, target) -- a completer

   if  torch.abs(input - target) == 0 then
      self.gradInput = torch.DoubleTensor({0})
   else
      self.gradInput = torch.sign(input - target)
   end

   return self.gradInput

end


-- ANALYZE PARAMETERS
-- ====================
if not opt then

   print '==> processing options'
   cmd = torch.CmdLine()

   cmd:text('Options:')

   cmd:option('-batchSize', 0, '0 : batch gradient, 1 : stochastic gradient, 2..n : batch')
   cmd:option('-posLab', '1', 'Label of the 1st class')
   cmd:option('-negLab', '-1', 'Label for the 2nd class, -1 means all other digits')
   cmd:option('-learningR', 1e-3, 'Value for Learning rate')
   cmd:option('-maxEpoch', 100, 'Maximum number of iterations')
   cmd:option('-criterion', 'mse', 'Loss function used, can be \'MSE\' or \'custom\'')

   opt = cmd:parse(arg or {})
end

posLab = parseLab(opt.posLab)
negLab = parseLab(opt.negLab)

-- CREATING MODEL 
-- ==============

local learning_rate= opt.learningR
local maxEpoch=opt.maxEpoch
local all_accTrain = {}
local all_accTest = {}
local all_losses = {}


local XTrain, yTrain, XTest, yTest = get(posLab,negLab)
local cardExemple, cardFeatures= X:size(1), X:size(2)

local model= nn.Linear(cardFeatures,1)
print("Linear Model with "..cardFeatures.." Features")
model:reset(0.1)

-- SELECTING CRITERION
-- ==================
opt.criterion = string.lower(opt.criterion)

if opt.criterion == 'mse' then
   criterion= nn.MSECriterion()
   print("Criterion used : MSE")
elseif opt.criterion == 'huber' then
   criterion=HuberLoss()
   print("Criterion used : HuberLoss")

else
   error("Bad criterion, can be 'mse' or 'huber'")
end

criterion = SeurinLoss()

-- SELECTING BATCH
-- ==================
local minibatchSize = opt.batchSize

if minibatchSize == 1 then
   gradType = 'stoch'
elseif minibatchSize == 0 then
   gradType = 'batch'
else
   gradType = 'minibatch'
end


-- LEARNING AND PREDICTING
-- ======================

print("Gradient used : "..gradType)

all_accTrain[1]= accuracy(model:forward(XTrain),yTrain)  --Remember accuracy on Train Dataset
print("Train :"..all_accTrain[1])
outputTest = model:forward(XTest)
all_accTest[1]= accuracy(outputTest,yTest)  --Remember accuracy on Test Dataset
print("Test :"..all_accTest[1])


timer = torch.Timer()

for iteration=2,maxEpoch do

   local loss=0
   local trackLoss = 0
   local tempAcc = 0

   if gradType == 'stoch' then

      for i = 1, cardExemple do

         local id = torch.random(cardExemple)
         local x, y = XTrain[id], yTrain[id]

         local output = model:forward(x)
         local loss = criterion:forward(output, y)

         trackLoss = trackLoss + loss

         local delta = criterion:backward(output, y)

         model:zeroGradParameters()
         model:backward(x, delta)
         model:updateParameters(learning_rate)
         
      end

      all_losses[iteration] = trackLoss/cardExemple

   elseif gradType == 'minibatch' then

      for i = 1, math.floor(cardExemple/minibatchSize) do

         local randomId=torch.randperm(cardExemple)
         randomId = randomId[{{1,minibatchSize}}]:long()
         
         local X,y = XTrain:index(1,randomId), yTrain:index(1,randomId)
         local output = model:forward(X)

         local delta = criterion:backward(output,y)

         model:zeroGradParameters()
         model:backward(X, delta)
         model:updateParameters(learning_rate)
         
      end

   elseif gradType == 'batch' then

      model:zeroGradParameters()
      
      local output = model:forward(XTrain)
      local loss = criterion:forward(output,yTrain)

      local delta = criterion:backward(output,yTrain)

      model:zeroGradParameters()
      model:backward(XTrain,delta)
      model:updateParameters(learning_rate)

   end

   all_accTrain[iteration]= accuracy(model:forward(XTrain),yTrain)  --Remember accuracy on Train Dataset

   
   outputTest = model:forward(XTest)
   all_accTest[iteration]= accuracy(outputTest,yTest)  --Remember accuracy on Test Dataset

   print("Train :"..all_accTrain[#all_accTrain])
   print("Test :"..all_accTest[#all_accTest])


   if iteration%10 == 0 then

      --plot(xs,ys,model,100)  -- uniquement si DIMENSION=2

      -- gnuplot.plot(torch.Tensor(all_accTrain))
      -- gnuplot.plot(torch.Tensor(all_accTest))

      gnuplot.plot({'Train', torch.Tensor(all_accTrain)},{'Test', torch.Tensor(all_accTest)})
      
      -- gnuplot.plot(torch.Tensor(all_losses))


   end
end

print("Train :"..all_accTrain[#all_accTrain])
print("Test :"..all_accTest[#all_accTest])
print("Temps Écoulé", timer:time().real)

require 'nn'
require 'gnuplot'
require 'io'
require 'tools'

------------ TOOLS : Accuracy/CreateDataset
-- =================================
local function accuracy(modelOutput, y)

   -- Matrix, return the whole accuracy
   modelOutput[torch.lt(modelOutput,0)] = -1 
   modelOutput[torch.gt(modelOutput,0)] = 1

   local resultMatrix = (torch.eq(modelOutput,y)):double()
   return resultMatrix:mean()

end

local function createDataset(cardPoints)
   local cardPoints = cardPoints or 1000

   local pos1 = torch.Tensor({1,1})
   local pos2 = torch.Tensor({-1,-1})

   local neg1 = torch.Tensor({-1,1})
   local neg2 = torch.Tensor({1,-1})

   local var_positive=1.0

   local X=torch.Tensor(cardPoints,2)
   local y=torch.Tensor(cardPoints,1)

   for i=1,cardPoints/4 do
      backwardIndex = cardPoints - i+1
      
      X[i]= pos1 + torch.randn(2)/1.5
      X[backwardIndex] = pos2 + torch.randn(2)/1.5
      y[i] = 1
      y[backwardIndex] = 1
   end

   for i=cardPoints/4+1,cardPoints/2 do
      backwardIndex = cardPoints - i+1
      X[i]= neg1 + torch.randn(2)/1.5
      X[backwardIndex] = neg2 + torch.randn(2)/1.5
      y[i] = -1
      y[backwardIndex] = -1
   end

   return X,y
end

-- ANALYZE PARAMETERS
-- ====================
if not opt then

   print '==> processing options'
   cmd = torch.CmdLine()

   cmd:text('Options:')

   cmd:option('-learningR', 1e-2, 'Value for Learning rate')
   cmd:option('-maxEpoch', 200, 'Maximum number of iterations')

   opt = cmd:parse(arg or {})
end

-- CREATING MODEL 
-- ==============

local learning_rate= opt.learningR
local maxEpoch=opt.maxEpoch
local all_accTrain = {}
local all_accTest = {}
local all_losses = {}
local criterion= nn.MSECriterion()

gradType = 'stoch'


XTrain, yTrain, XTest, yTest = createDataset(1000)
plot_points(XTrain,yTrain)

cardExemple, cardFeatures= XTrain:size(1), XTrain:size(2)

print(cardFeatures,cardExemple)

nhidden = cardFeatures+4


--With nn.sequential
print("With sequential :")

model = nn.Sequential()
model:add(nn.Linear(2,nhidden))
model:add(nn.Tanh())

model:add(nn.Linear(nhidden,1))
model:add(nn.Tanh())

for i=1,maxEpoch do

   local output = model:forward(XTrain)
   local err = criterion:forward(output, yTrain)

   model:zeroGradParameters()
   local grad = criterion:backward(output, yTrain)
   model:backward(XTrain, grad)
   model:updateParameters(0.1)

end


output = model:forward(XTrain)
print("Accuracy :",accuracy(output,yTrain))  --Remember accuracy on Train Dataset



local model = {}
model[1]= nn.Linear(cardFeatures,nhidden)
model[2]= nn.Tanh()
--rmodel[2].updateGradInput = function() end

model[3]= nn.Linear(nhidden,1)
model[4]= nn.Tanh()
--model[4].updateGradInput = function() end


print("Linear Model with "..cardFeatures.." Features")
model[1]:reset(0.1)
model[2]:reset(0.1)
model[3]:reset(0.1)
model[4]:reset(0.1)


-- LEARNING AND PREDICTING
-- ======================

timer = torch.Timer()

for iteration=1,maxEpoch do

   local loss=0
   local trackLoss = 0
   local tempAcc = 0

   for i = 1, cardExemple do

      local tempInputList = {}

      local id = torch.random(cardExemple)
      local x, y = XTrain[id], yTrain[id]
      table.insert(tempInputList,x)

      -- Forward for the whole model
      local output = model[1]:forward(x)
      table.insert(tempInputList,output)

      for numComponent=2,#model do
         output = model[numComponent]:forward(output)
         table.insert(tempInputList,output)
      end

      --
      local loss = criterion:forward(output, y)
      trackLoss = trackLoss + loss

      model[1]:zeroGradParameters()
      model[2]:zeroGradParameters()
      model[3]:zeroGradParameters()
      model[4]:zeroGradParameters()

      local delta = criterion:backward(tempInputList[5], y)
      -- Compute update parameters (model:backward for every components)
      for numElem=1,#model do
         backwardElem = #model+1-numElem
         delta = model[backwardElem]:backward(tempInputList[backwardElem], delta)
         model[backwardElem]:updateParameters(learning_rate)
      end
   end


   all_losses[iteration] = trackLoss/cardExemple

   

   output = model[1]:forward(XTrain)
   for numElem=2,#model do
      output = model[numElem]:forward(output)
   end

   all_accTrain[iteration]= accuracy(output,yTrain)  --Remember accuracy on Train Dataset

   --   outputTest = model:forward(XTest)
   --   all_accTest[iteration]= accuracy(outputTest,yTest)  --Remember accuracy on Test Dataset


   if iteration%10 == 0 then


      gnuplot.plot(torch.Tensor(all_accTrain))
      -- gnuplot.plot(torch.Tensor(all_accTest))

      --gnuplot.plot({'Train', torch.Tensor(all_accTrain)},{'Test', torch.Tensor(all_accTest)})
      -- gnuplot.plot(torch.Tensor(all_losses))
   end
end


print("A la mano nn :")
print("Accuracy :"..all_accTrain[#all_accTrain])
--print("Test :"..all_accTest[#all_accTest])
print("Temps Écoulé", timer:time().real)



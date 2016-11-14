require 'nn'
require 'gnuplot'
require 'io'

------------ TOOLS : Accuracy/CreateDataset
-- =================================
local function accuracy(modelOutput, y)

   -- Matrix, return the whole accuracy
   modelOutput[torch.lt(modelOutput,0)] = -1 
   modelOutput[torch.gt(modelOutput,0)] = 1

   local resultMatrix = (torch.eq(modelOutput,y)):double()
   return resultMatrix:mean()

end

local function createDatasetLinear(cardPoints)
   local cardPoints = cardPoints or 1000

   local pos = torch.Tensor({1,1})
   local neg = torch.Tensor({-1,-1})

   local var_positive=1.0

   local X=torch.Tensor(cardPoints,2)
   local y=torch.Tensor(cardPoints,1)

   for i=1,cardPoints/2 do
      
      X[i]= pos + torch.randn(2)/1.5
      y[i] = 1
   end

   for i=cardPoints/2+1,cardPoints do
      X[i]= neg + torch.randn(2)/1.5
      y[i] = -1
   end

   return X,-y
end


local function createDatasetNonLinear(cardPoints)
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
   cmd:option('-model', 1, '1:MLP torch, 2:ReQU 3:Linear Torch 4:LinearCustom 5:MLP Custom')
   cmd:option('-dataset', 1, '1:Linear 2::nonLinear')

   opt= cmd:parse(arg or {})
end

-- CREATING MODEL 
-- ==============

local learning_rate= opt.learningR
local maxEpoch=opt.maxEpoch
local all_accTrain = {}
local all_accTest = {}
local all_losses = {}
local criterion= nn.MSECriterion()
local modelType = opt.model



-- CUSTOM MODULE REQU and LINEAR
-- =============================
local ReCustom, parent = torch.class('ReQU', 'nn.Module')

function ReQU:__init(inputSize, outputSize)
   parent.__init(self)
end

function ReQU:updateOutput(input)
   input[torch.le(input,0)] = 0
   self.output = torch.cmul(input, input)
   return self.output
end

function ReQU:accGradParameters(input, gradOutput)
   --No parameters, no update
   return self
end

function ReQU:updateGradInput(input,gradOutput)
   -- Dérivée de l'erreur par rapport à un xi
   input[torch.le(input,0)] = 0
   self.gradInput = torch.cmul(gradOutput,input)*2
   return self.gradInput
end

-- =================
local customLinear_, parent = torch.class('customLinear', 'nn.Module')

function customLinear:__init(inputSize, outputSize)
   parent.__init(self)
   self.weight = 2*(torch.Tensor(outputSize, inputSize):uniform()-0.5)
   self.inputSize = inputSize
   self.outputSize = outputSize

   self.bias = 1
   self.gradUpdate = torch.Tensor(outputSize,inputSize):fill(0)
   self.gradInput = torch.Tensor(inputSize):fill(0)
   
end

function customLinear:updateOutput(input)

   if input:size(1)~=self.weight:size(2) then
      input = input:t()
   end

   self.output = self.weight*input
   self.output = self.output +self.bias

   return self.output
end


function customLinear:zeroGradParameters()
   self.gradUpdate:zero()
   self.gradInput:zero()
   self.biasUp = 0
end

function customLinear:updateParameters(learningRate)
   self.weight = self.weight - learningRate*self.gradUpdate
   self.bias = self.bias - learningRate*self.biasUp
   
   self:zeroGradParameters()
   return self.weight
end


function customLinear:accGradParameters(input, gradOutput)

   if input:size(1)==1 then
      input = input[{1,{}}]
   else
      input = input[{{},1}]
   end

   if gradOutput:size(1)==1 then
      gradOutput = gradOutput[{1,{}}]
   else
      gradOutput = gradOutput[{{},1}]
   end
   self.gradUpdate = self.gradUpdate:addr(gradOutput,input)
   self.biasUp = torch.sum(gradOutput)
end

function customLinear:updateGradInput(input,gradOutput)

   if gradOutput:size(1)==1 then
      gradOutput = gradOutput[{1,{}}]
   elseif gradOutput:size(2)==1 then
      gradOutput = gradOutput[{{},1}]
   else
      error("WTF dude")
   end

   self.gradInput:addmv(self.weight:transpose(1,2), gradOutput)
   return self.gradInput

end

if opt.dataset == 1 then
   XTrain, yTrain, XTest, yTest = createDatasetLinear(1000)
elseif opt.dataset == 2 then
   XTrain, yTrain, XTest, yTest = createDatasetNonLinear(1000)
else
   error('Wrong Dataset dude')
end

local cardExemple, cardFeatures= XTrain:size(1), XTrain:size(2)
local nhidden = cardFeatures+4
local model = nn.Sequential()


if modelType == 1 then -- MLP torch : Linear -> Tanh -> Linear Tahn
   print("Using TORCH MLP")

   model:add(nn.Linear(cardFeatures,nhidden))
   model:add(nn.Tanh())
   
   model:add(nn.Linear(nhidden,1))
   model:add(nn.Tanh())

elseif modelType == 2 then -- MLP torch Custom ReQU: Linear -> reQU -> Tanh -> Linear -> Tahn
   print("Using custom ReQU")

   model:add(nn.Linear(cardFeatures,nhidden))
   model:add(ReQU())
   model:add(nn.Tanh())
   
   model:add(nn.Linear(nhidden,1))
   model:add(nn.Tanh())

elseif modelType == 3 then -- Linear : Linear -> Tanh

   model:add(nn.Linear(cardFeatures,1))
   model:add(nn.Tanh())

   
elseif modelType == 4 then -- Linear Custom : customLinear -> Tanh
   print("Using Custom Linear")
   model:add(customLinear(cardFeatures,1))
   model:add(nn.Tanh())

elseif modelType == 5 then -- MLP Custom
   print("Using Custom MLP")

   model:add(customLinear(cardFeatures,nhidden))
   model:add(nn.Tanh())
   
   model:add(customLinear(nhidden,1))
   model:add(nn.Tanh())

end

output = model:forward(XTrain)
all_accTrain[1]= accuracy(output,yTrain)  --Remember accuracy on Train Dataset
print("Accuracy on init :",all_accTrain[1])

for i=1,maxEpoch do
   
   for ex = 1, cardExemple do

      local tempInputList = {}

      local id = torch.random(cardExemple)
      local x, y = XTrain[{{id}}], yTrain[{{id}}]

      local output = model:forward(x)
      local err = criterion:forward(output, y)

      model:zeroGradParameters()
      local grad = criterion:backward(output, y)
      model:backward(x, grad)
      model:updateParameters(learning_rate)
   end

   output = model:forward(XTrain)
   all_accTrain[i+1]= accuracy(output,yTrain)  --Remember accuracy on Train Dataset

   if i%30 == 0 then
      gnuplot.plot(torch.Tensor(all_accTrain))
   end

end

print("Final Accuracy : ", all_accTrain[#all_accTrain])
--print("Test :"..all_accTest[#all_accTest])



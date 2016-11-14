require 'nn'
require 'nngraph'
require 'gnuplot'
require 'io'
require 'tools'

--[[GRAPH NN
    =========
La régularisation L2 permet de minimiser les poids W d'une unité linéaire.
Un problème est qu'elle n'impose jamais que les poids soit complètement à zéro.

Le but de la régularisation L1 est d'avoir un ensemble de poids W qui est sparse.

Pour implémenter ceci sous Torch avec le module 'Linear', il va falloir faire les choses
différemment car on ne peut connecter directement le module L1 Penalty sur le module 
Linear car celui-ci agirait sur les sorties du module linéaire,
 or on aimerait minimiser et mettre à zéro certains poids, on va donc agir sur les entrées

C'est pour cela que l'on va faire ajouter un module linéaire avant 
les entrées et une unité 'L1Penalty', si la variable est intéressante, alors elle aura un poids positif,
sinon il sera à 0 ou négatif (l'unité ReLU derrière passera les négatifs à zéro)


]]--

-- ANALYZE PARAMETERS
-- ====================
if not opt then

   cmd = torch.CmdLine()
   cmd:text('Options:')

   cmd:option('-learningR', 1e-2, 'Value for Learning rate')
   cmd:option('-maxEpoch', 100, 'Maximum number of iterations')
   cmd:option('-dataset', 1, '1:Linear 2:nonLinear')
   cmd:option('-lambda', 1e-2, 'L1 regularization Factor')

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
local lambda = opt.lambda


if opt.dataset == 1 then
   XTrain, yTrain = get({1},{-1})
elseif opt.dataset == 2 then
   XTrain, yTrain, XTest, yTest = createDatasetLinear(1000)
elseif opt.dataset == 3 then
   XTrain, yTrain, XTest, yTest = createDatasetNonLinear(1000)
else
   error('Wrong Dataset dude')
end


local cardExemple, cardFeatures= XTrain:size(1), XTrain:size(2)
--Creating graph model===============
--==================================

m1 = nn.Linear(1, cardFeatures)
m2 = nn.L1Penalty(lambda)
m3 = nn.ReLU()
m4 = nn.CMulTable()
m5 = nn.Linear(cardFeatures, 1)
m6 = nn.Tanh()
c = nn.MSECriterion()

-- forward
-- o1 = m1:forward(torch.Tensor({1}))
-- o2 = m2:forward(o1)
-- o3 = m3:forward(o2)
-- o4 = m4:forward({x, o3})
-- o5 = m5:forward(o4)
-- o6 = m6:forward(o5)
-- loss = c:forward(o6, y)

-- dl = c:backward(o6, y)
-- d6 = m6:backward(o5, dl)
-- d5 = m5:backward(o4, d6)
-- d4 = m4:backward({x, o3}, d5)
-- d3 = m3:backward(o2, d4[2])
-- d2 = m2:backward(o1, d3)
-- d1 = m1:backward(torch.Tensor({1}), d2)
-- m5:updateParameters(learning_rate)


input1 = nn.Identity()()
input2 = nn.Identity()()

m1 = nn.Linear(1, cardFeatures)(input2)
m2 = nn.L1Penalty(lambda)(m1)
m3 = nn.ReLU()(m2)
m4 = nn.CMulTable()({input1, m3})
m5 = nn.Linear(cardFeatures, 1)(m4)
output = nn.Tanh()(m5)

model = nn.gModule({input1, input2}, {output})
--graph.dot(model.fg, 'Example')

-- for indexNode, node in ipairs(model.forwardnodes) do
--    if node.data.module then
--       print(indexNode, node.data.module)
--    end
-- end

local w = model.forwardnodes[11].data.module.weight
print("Number of weight set to zeros :", numZeros(w))


--======================================================
inRelu = torch.Tensor({1})
inReluMat = torch.ones(XTrain:size(1),1)

output = model:forward({XTrain,inReluMat})
all_accTrain[1]= accuracy(output,yTrain)  --Remember accuracy on Train Dataset
print("Accuracy on init :",all_accTrain[1])


for i=1,maxEpoch do
   

   local id = torch.random(cardExemple)
   local x, y = XTrain, yTrain

   local output = model:forward({x,inReluMat})
   local err = criterion:forward(output, y)

   model:zeroGradParameters()
   local grad = criterion:backward(output, y)
   model:backward({x,inRelu}, grad)
   model:updateParameters(learning_rate)

   output = model:forward({XTrain,inReluMat})
   all_accTrain[i+1]= accuracy(output,yTrain)  --Remember accuracy on Train Dataset


   if i%5 == 0 then
      gnuplot.plot(torch.Tensor(all_accTrain))
   end
   
end

print("Final Accuracy : ", all_accTrain[#all_accTrain])
local output = model:forward({XTrain[1],inRelu})


local w = model.forwardnodes[9].data.module.output
print("Number of weight set to zeros :", numZeros(w))


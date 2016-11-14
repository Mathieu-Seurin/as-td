local mnist = require 'mnist'


------------ TOOLS : Accuracy/CreateDataset
-- =================================
function accuracy(modelOutput, y)

   -- Matrix, return the whole accuracy
   modelOutput[torch.lt(modelOutput,0)] = -1 
   modelOutput[torch.gt(modelOutput,0)] = 1

   local resultMatrix = (torch.eq(modelOutput,y)):double()
   return resultMatrix:mean()

end

function createDatasetLinear(cardPoints)
   local cardPoints = cardPoints or 1000

   local pos = torch.Tensor({1,1})
   local neg = torch.Tensor({-1,-1})

   local var_positive=1.0

   local X=torch.Tensor(cardPoints,3)
   local y=torch.Tensor(cardPoints,1)

   for i=1,cardPoints/2 do
      
      X[{i,{1,2}}]= pos + torch.randn(2)/1.5
      X[{i,3}] = 1
      y[i] = 1
   end

   for i=cardPoints/2+1,cardPoints do
      X[{i,{1,2}}]= neg + torch.randn(2)/1.5
      X[{i,3}] = 1
      y[i] = -1
   end

   return X,-y
end


function createDatasetNonLinear(cardPoints)
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

function numZeros(w)
   local mask = torch.eq(w,0):double()
   return mask:sum()
end

-- ==============================================================
-- ================== Load MNIST ================================
-- ==============================================================


-- utilisation :
--   load_mnist = require 'load_mnist'
--   xtrain,ytrain=load_mnist.get_train(2,3)
--  xtest,test=load_mnist.get_test(2,3)
--  xtrain,ytrain,xtest,ytest = load_mnist.get(2,3)
--  gnuplot.imagesc(xtrain[1]:reshape(28,28))

local train = mnist.traindataset()
local test = mnist.testdataset()
local full_train = torch.div(torch.reshape(train.data,train.data:size(1),28*28):double(),torch.max(train.data))
local full_test = torch.div(torch.reshape(test.data,test.data:size(1),28*28):double(),torch.max(train.data))

-- gnuplot.imagesc(train.data[1])
--print(train.label[1])
--get_idx(1,y)


local function get_idx(lab,labels,except)

   local idx = torch.linspace(1,labels:size(1),labels:size(1)):long()

   if type(lab) ~= 'table' then
      error("Label have to be a table, not a "..type(lab))

   elseif #lab == 1 then
      lab = lab[1]
      if lab == -1 then
         if not except then error("Need labpos to be specified") end

         if #except == 1 then
            return idx[labels:ne(except[1])]
         else
            idBool = torch.ne(labels, except[1])
            for digit=2,#except do
               --WARNING CRYPTIC FORMULA :
               -- Since i can't find a 'and' function for two Byte Tensor, i tweak with this :
               -- if a+b > 1.5  equivalent for logical 'and'
               idBool = torch.gt(idBool+torch.ne(labels, except[digit]),1.5)
            end
         end
         return idx[idBool]
         
      else
         return idx[labels:eq(lab)]
      end

   elseif #lab >= 2 then
      idBool = torch.eq(labels, lab[1])
      for label=2,#lab do
         --WARNING CRYPTIC FORMULA :
         -- Since i can't find a 'or' function for two Byte Tensor, i tweak with this :
         -- if a+b > 0.5 equivalent for logical 'or'
         idBool = torch.gt(idBool+torch.eq(labels, lab[label]), 0.5)
      end
      return idx[idBool]

   else
      error("How did you get there ?")
   end
end

-- utilisation : par exemple pour recuperer les labels 2 et 5,  get_subsets({2},{5},full_train,train.label)
local function get_subsets(labpos,labneg,data,labels)
   local idxpos = get_idx(labpos,labels)
   local idxneg = get_idx(labneg,labels,labpos) --if labneg == '-1', will retreive all digit except labpos   
   local xset = torch.cat(data:index(1,idxpos),data:index(1,idxneg),1)
   local yset = torch.cat(torch.ones(idxpos:size(1),1),torch.ones(idxneg:size(1),1):fill(-1),1)
   local idx = torch.randperm(yset:size(1)):long()
   return xset:index(1,idx),yset:index(1,idx)
end

local function get_train(labpos,labneg)

   return get_subsets(labpos,labneg,full_train,train.label)
end

local function get_test(labpos,labneg)
   return get_subsets(labpos,labneg,full_test,test.label)
end

function get(labpos,labneg)
   X,y = get_train(labpos,labneg)
   xTest, yTest = get_test(labpos,labneg)
   return X,y,xTest,yTest
end

require 'nn'
require 'loadData'
require 'io'
require 'cunn'


function predictUsingWave(predNet,sizeSeq,nInput,nOutput,numLayer)
   
   data = loadData('paco.mat')
   batches=splitDataBatch(data,nInput,nOutput,numLayer,1)

   baseSeq = batches[1]['x']

   numFeat = 128
   generatedSeq = {}

   softmax = nn.SoftMax()

   for t = 1,sizeSeq do
      vectOut = predNet:forward(baseSeq)

      -- Getting last element
      vectOut = softmax:forward(vectOut[{1,vectOut:size(2),{}}])
      
      pred = torch.cumsum(vectOut)
      chosen = 1
      gen = torch.rand(1)[1]
      while gen > pred[chosen] do chosen = chosen+1 end

      note = torch.zeros(numFeat)
      note[chosen] = 1

      generatedSeq[#generatedSeq+1] = chosen
      
      baseSeq[{1,{1,nInput-1},{}}] = baseSeq[{1,{2,nInput},{}}]
      baseSeq[{1,nInput,{}}] = note

   end

   print("seq : ", generatedSeq)
   matio.save('generatedSeq.mat',torch.Tensor(generatedSeq))

end

local nOutput = 19
local numLayer = 2
local nInput = 21

local filterSz = 2
local sizeSeq = 256

net = torch.load(MODELDIR..'WaveNet'..numLayer..'lay'..filterSz..'.model')

predictUsingWave(net,sizeSeq,nInput,nOutput,numLayer)


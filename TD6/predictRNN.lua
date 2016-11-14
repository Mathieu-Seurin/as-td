require 'nn'
require 'nngraph'
require 'creationRNN'
require 'load_exemple'
require 'io'

function predictUsingRNN(predNet, ht, vocab, sizeSeq)
   seq = ''
   letter = torch.zeros(35)
   letter[10] = 1

   for t = 1,sizeSeq do
      gen = torch.rand(1)[1]
      vectPred = predNet:forward({ht,letter})
      out = vectPred[1]
      ht = vectPred[2]

      pred = torch.cumsum(out)
      chosen = 1
      while gen > pred[chosen] do chosen = chosen+1 end
      seq = seq..vocab[chosen]
   end

   print("seq : ".. seq)
end

local N = 20
local vocab = torch.load('vocab.t7')

net = torch.load('lovecraftRNN.pred'..N)
ht = torch.load('lovecraftRNN.ht'..N)
--ht = torch.zeros(N)
if ht:dim() ~= 1 then
   ht = ht[ht:size(1)]
end

predictUsingRNN(net, ht, inverseVoc(vocab), 25)

require 'nn'
require 'nngraph'
require 'creationRNN'
require 'load_exemple'
require 'io'

function predictUsingRNN(predNet, vocab, sizeSeq, N)
   seq = ''
   letter = torch.zeros(35)
   letter[6] = 1
   ht = torch.zeros(N)

   for t = 1,sizeSeq do
      gen = torch.rand(1)[1]
      vectOut = predNet:forward({ht,letter})
      out = vectOut[1]
      ht = vectOut[2]
      
      pred = torch.cumsum(out)
      chosen = 1
      while gen > pred[chosen] do chosen = chosen+1 end
      seq = seq..vocab[chosen]

      letter = torch.zeros(35)
      letter[chosen] = 1
   end

   print("seq : ".. seq)
end

local N = 150
local T = 150
local vocab = torch.load('vocab.t7')

net = torch.load('lovecraftRNN.pred'..N)
predictUsingRNN(net, inverseVoc(vocab), T, N)

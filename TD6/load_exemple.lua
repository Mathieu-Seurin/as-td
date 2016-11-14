require 'torch'
local CharLMMinibatchLoader=require 'CharLMMinibatchLoader'

function splitBatch(batch_size, seq_length)
   local v=CharLMMinibatchLoader.create("data.t7","vocab.t7",batch_size,seq_length)
   return v
end

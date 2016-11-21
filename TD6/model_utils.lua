require 'torch'


--======== TOOLS TO INTROSPECT AND LOAD
function reformatBatch(batchRaw,M,N,batchSize,T)

   batchRaw = batchRaw:double():t()
   batch = {}

   --INIT or continue computing
   batch[1] = torch.zeros(batchSize,N)
   
   --Reformat batch
   for numT = 1,T do
      temp = torch.zeros(batchSize,M)
      for di = 1,batchSize do
         temp[di][batchRaw[numT][di]] = 1
      end
      batch[numT+1] = temp
   end

   batchRaw = batchRaw[{{2,T}}]
   
   -- local iter = {}
   -- for count=1,M do
   --    iter[count] = count
   -- end
   -- iter = torch.Tensor(iter)

   -- batchTrans = {0}
   -- for item=2,N+1 do
   --    gt = torch.gt(batch[item],0)
   --    batchTrans[item] = iter[gt][1]
   -- end

   -- batchTrans = torch.Tensor(batchTrans)
   
   -- print("batchTrans",batchTrans)
   -- print("batchRaw",batchRaw)


   
   return batch,batchRaw
end

function inverseVoc(vocab)
   inverseVoca = {}
   for i,item in pairs(vocab) do
      inverseVoca[item] = i
   end
   return inverseVoca
end

function printBatch(batchRaw,vocab)
   inVoc = inverseVoc(vocab)
   str = ''
   for i,letter in ipairs(batchRaw[1]:totable()) do
      str = str..inVoc[letter]
   end
   print(str)
end   

function generateRandomY(N,M,batchSize)
   local yRand = {}
   for t=1,N-1 do
      yRand[t] = torch.rand(batchSize,M)
      local div = yRand[t]:sum(2)
      div = div:view(div:nElement())
      local outer = torch.zeros(batchSize,M)
      div = outer:addr(div, torch.ones(M) )
      yRand[t] = yRand[t]:cdiv(div)
   end

   return yRand
end

function introspect_parameters(net)
   for indexNode, node in ipairs(net.forwardnodes) do
      -- if node.data.module then
      --    print(node.data.module,indexNode)
      -- end

      if indexNode == 7 then
         for indexSub, sub in ipairs(node.data.module.forwardnodes) do
            -- if sub.data.module then
            --    print(sub.data.module, indexSub)
            if indexSub == 16 then
               print(sub.data.module:parameters()[1], indexNode)
            end
         end
      end
   end
end


function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then
      io.close(f)
      return true
   else return false
   end
end

function loadCritRNN(T)
   c = nn.ParallelCriterion()
   for t = 1,T-1 do
      crit = nn.CrossEntropyCriterion()
      crit.nll.sizeAverage = false
      c:add(crit)
   end
   return c

end


local osef, parent = torch.class('introLin', 'nn.Linear')

function introLin:__init(inputSize,outputSize)
   parent.__init(self,inputSize,outputSize)
end

function introLin:updateOutput(input)
   -- CUSTOM CLASS TO INTROSPECT NNGRAPH """"EASILY"""""
   print(input)
   
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      if self.bias then self.output:copy(self.bias) else self.output:zero() end
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      updateAddBuffer(self, input)
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      if self.bias then self.output:addr(1, self.addBuffer, self.bias) end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

-- From GITHUB, https://github.com/eladhoffer/recurrent.torch/blob/master/models/GRU.lua
-- for testing purpose to check if my implementation is drunk or just acting normally
function create_GRU(inputSize, outputSize, initWeights)
   -- there will be 2 input: {input, state}
   local initWeights = initWeights or 0.08
   local input = nn.Identity()()
   local state = nn.Identity()()

   -- GRU tick
   -- forward the update and reset gates
   local input_state = nn.JoinTable(1,1)({input, state})
   local gates = nn.Sigmoid()(nn.Linear(inputSize + outputSize, 2*outputSize)(input_state))
   local update_gate, reset_gate = nn.SplitTable(1,2)(nn.Reshape(2, outputSize)(gates)):split(2)

   -- compute candidate hidden state
   local gated_hidden = nn.CMulTable()({reset_gate, state})
   local input_gHidden = nn.JoinTable(1,1)({input, gated_hidden})
   local hidden_candidate = nn.Tanh()(nn.Linear(inputSize + outputSize, outputSize)(input_gHidden))
   -- compute new interpolated hidden state, based on the update gate
   local zh = nn.CMulTable()({update_gate, hidden_candidate})
   local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), state})
   local nextState = nn.CAddTable()({zh, zhm1})

   local rnnModule = nn.gModule({state,input}, {nextState})
   rnnModule:apply( --initialize recurrent weights
      function(m)
         if m.weight then
            m.weight:uniform(-initWeights, initWeights)
         end
         if m.bias then
            m.bias:uniform(-initWeights, initWeights)
         end
      end
   )

   -- graph.dot(rnnModule.fg, 'GRU', 'GITGRUFG')
   return rnnModule

end


function clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end


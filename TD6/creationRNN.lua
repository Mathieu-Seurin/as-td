require 'nn'
require 'nngraph'
require 'model_utils'

--======== ACTUAL MODEL
-- GRU, D, whole RNN and Criterion
function create_RNN(M,N,T)
   
   --==== CREATING GRU
   --========================

   local ht_1 = nn.Identity()():annotate{name='IN 1'}
   local x = nn.Identity()():annotate{name='IN 2'}
   
   local Un = nn.Linear(N,N)(ht_1):annotate{name='Un'}
   local Uz = nn.Linear(N,N)(ht_1):annotate{name='Uz'}
   
   local Wz = nn.Linear(M,N)(x):annotate{name='Wz'}
   local Wn = nn.Linear(M,N)(x):annotate{name='Wn'}
   
   local sz = nn.CAddTable()({Uz,Wz}):annotate{name='sz'}
   local sr = nn.CAddTable()({Un,Wn}):annotate{name='sr'}
   
   local rt = nn.Sigmoid()(sr)
   local zt= nn.Sigmoid()(sz)
   
   local rt2 = nn.CMulTable()({rt,ht_1}):annotate{name='rt2'}
   local rt3 = nn.Linear(N,N)(rt2):annotate{name='rt3'}
   
   local hthat = nn.Tanh()(nn.CAddTable()({rt3,nn.Linear(M,N)(x):annotate{name='Hthat'}}))
   
   local one_zt = nn.AddConstant(1)(nn.MulConstant(-1)(zt))
   local ht = nn.CAddTable()({nn.CMulTable()({hthat,zt}),nn.CMulTable()({ht_1,one_zt})})
   
   nngraph.annotateNodes()
   local GRU = nn.gModule({ht_1,x},{ht})
   --graph.dot(GRU.fg, 'GRU', 'MYGRUFG')
   
   --ht = GRU:forward({torch.rand(N),torch.rand(M)})

   --==== CREATING D function
   --========================

   local in_d = nn.Identity()()
   local out_d = nn.Linear(N,M)(in_d)
   
   local d = nn.gModule({in_d},{out_d})

   --res_test = d:forward(ht)

   --==== CREATING Network
   --========================
   local gs = clone_many_times(GRU,T+1)
   local ds = clone_many_times(d,T+1)

   local inputs = {}
   local outputs = {}
   local z = nn.Identity()():annotate{name='Z ZE FIRST'}
   inputs[1] = z

   for t = 1,T do
      inputs[t+1] = nn.Identity()():annotate{name='input'..t} -- (xt)
      z = gs[t]({z,inputs[t+1]}):annotate{name='GRU'..t} -- zt+1 = g(zt, xt+1)
      outputs[t] = ds[t](z):annotate{name='DECODE'..t}
   end

   local model = nn.gModule(inputs,outputs)


   --===== Prediction graph
   --========================
   -- Adding a softmax at the end
   -- to get a probability vector
   
   local x = nn.Identity()()
   local st = nn.Identity()()
   
   local zpred = gs[T+1]({st,x}) --zt+1 = g(zt, xt+1)
   local outpred = nn.SoftMax()(ds[T+1](zpred))   -- Adding Softmax to predict proba distribution
   
   local netPred = nn.gModule({st,x},{outpred,zpred})
   --ytest = netPred:forward({ht,res_test})

   --======= Criterion ======
   --========================

   local c = loadCritRNN(T)
   
   return model,c, netPred
end

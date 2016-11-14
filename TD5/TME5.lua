require 'nn'
require 'nngraph'
--hide

lambda = 10e-1
learning_rate = 10e-1
dim = 2

m1 = nn.Linear(1, dim)
m2 = nn.L1Penalty(lambda)
m3 = nn.ReLU()
m4 = nn.CMulTable()
m5 = nn.Linear(dim, 1)
m6 = nn.Tanh()
c = nn.MSECriterion()

x = torch.rand(2)
y = torch.Tensor({1})

-- forward
o1 = m1:forward(torch.Tensor({1}))
o2 = m2:forward(o1)
o3 = m3:forward(o2)
o4 = m4:forward({x, o3})
o5 = m5:forward(o4)
o6 = m6:forward(o5)
loss = c:forward(o6, y)

dl = c:backward(o6, y)
d6 = m6:backward(o5, dl)
d5 = m5:backward(o4, d6)
d4 = m4:backward({x, o3}, d5)
d3 = m3:backward(o2, d4[2])
d2 = m2:backward(o1, d3)
d1 = m1:backward(torch.Tensor({1}), d2)
m5:updateParameters(learning_rate)


input1 = nn.Identity()()
input2 = nn.Identity()()
m1 = nn.Linear(1, dim)(input2)
m2 = nn.L1Penalty(lambda)(m1)
m3 = nn.ReLU()(m2)
m4 = nn.CMulTable()({input1, m3})
m5 = nn.Linear(dim, 1)(m4)
output = nn.Tanh()(m5)


model = nn.gModule({input1, input2}, {output})
hat = model:updateOutput({x, torch.Tensor({1})})
model:updateGradInput({x, torch.Tensor({1})}, y)
graph.dot(model.fg, 'Example', 'graphCrilout')





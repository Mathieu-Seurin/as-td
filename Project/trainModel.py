# coding: utf-8
from tools import *

from torch import optim

numEpoch = 10
sizeBatch = 2
learningRate = 1e-2

numLayer = 2

filterSize = 2
dilation = False

if not dilation:
    nInput = (filterSize-1)*numLayer + 1
else:
    assert False,"NotNow"

data = loadData("dummy.mat")

numFeatures = np.size(data,1)

numBatches = int(np.floor(np.size(data,0)/nInput/sizeBatch))

net = WaveNet(numFeatures,numLayer,filterSize,dilation)
crit = createCriterion()

optimizer = optim.SGD(net.parameters(), lr = learningRate)

print """Begin Training :
Size of batch : {}
Number of batch : {}
Number of Input : {}""".format(sizeBatch,numBatches,nInput)

for epoch in range(numEpoch):
    for x,y in genBatches(data, nInput, sizeBatch, numFeatures):

        optimizer.zero_grad()
        yhat = net(x)
        loss = crit(yhat, y)
        print("Loss :",loss)
        loss.backward()
        optimizer.step()



numTest = 100
accuracy = 0
for i,(x,y) in enumerate(genBatches(data, nInput, 1, numFeatures)):
    if i >= numTest:
        break

    yhat = net(x)
    _, pred = torch.max(yhat.data,1)
    if pred[0][0]==y.data[0]:
        accuracy +=1

print "Test : {}% accuracy".format(100*accuracy/numTest)
    
    
    
    

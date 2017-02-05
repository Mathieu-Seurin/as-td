# coding: utf-8
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.io as sio


class __WaveNet(nn.Module):

    def __init__(self, numFeatures, numLayer, kernelSize, dilation=False):

        super(WaveNet, self).__init__()

        assert kernelSize==2,"Not sure about behavior with filter size different than 2"
        assert dilation==False,"Not sure about behavior with dilation"

        self.dilation = dilation

        self.numFeatures = numFeatures
        self.numLayer = numLayer
        self.kernelSize = kernelSize

        sizeIn = self.numFeatures
        sizeOut = self.numFeatures
        kernelSize = self.kernelSize
        stride = kernelSize-1
        if self.dilation:
            dilation = i*2
        else:
            dilation = 0

        self.conv = []
        for i in range(2):
            self.conv.append(nn.Conv1d(sizeIn,sizeOut,kernelSize,stride,dilation))

    def parameters(self):
        params = []
        for lay in self.conv:
            params += list(lay.parameters())
        return params
        
    def forward(self, x):

        x = F.relu(self.conv[0](x))
        x = self.conv[1](x)
        
        x = torch.squeeze(x,2)
        return x


class WaveNet(nn.Module):

    def __init__(self, numFeatures, numLayer, kernelSize, dilation=False):

        super(WaveNet, self).__init__()

        assert kernelSize==2,"Not sure about behavior with filter size different than 2"
        assert dilation==False,"Not sure about behavior with dilation"

        self.dilation = dilation

        self.numFeatures = numFeatures
        self.numLayer = numLayer
        self.kernelSize = kernelSize

        sizeIn = self.numFeatures
        sizeOut = self.numFeatures
        kernelSize = self.kernelSize
        stride = kernelSize-1
        if self.dilation:
            dilation = i*2
        else:
            dilation = 0

        self.convList = []
        for i in range(self.numLayer):
            self.convList.append(nn.Conv1d(sizeIn,sizeOut,kernelSize,stride,dilation))
            
    def parameters(self):
        allParams = []
        for lay in self.convList:
            allParams += list(lay.parameters())
        return allParams
        

    def forward(self, x):

        for i in range(self.numLayer):
            if i<self.numLayer-1:
                x = F.relu(self.convList[i](x))
            else:
                x = self.convList[i](x)
                x = torch.squeeze(x,2)
        return x


def createCriterion():
    crit = nn.CrossEntropyLoss()
    return crit

    
def loadData(fileName):

    if fileName[-4:] == '.mat':
        seq = sio.loadmat(fileName)['mat']
    else:
        seq = sio.loadmat(fileName+'.mat')['mat']
    return seq
    
        
def genBatches(data, nInput, sizeBatch, numFeatures, stoch=True):
    
    numEx = np.size(data,0)
    numBatch = int(np.floor(numEx/nInput/sizeBatch))

    for b in range(numBatch):
        x = torch.zeros(sizeBatch,numFeatures,nInput)
        y = torch.zeros(sizeBatch).long()

        for s in range(sizeBatch):
            if stoch:
                select=np.random.randint(0,numEx-nInput)
            else:
                select = 0
                
            yBeginIndex = select+nInput

            x[s,:] = torch.from_numpy(data[select:select+nInput,:].T)
            y[s] = np.where(data[yBeginIndex])[0][0]
            
        yield Variable(x,requires_grad=True),Variable(y)

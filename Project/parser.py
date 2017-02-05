#coding: utf-8
import music21 as ms
import numpy as np
import scipy.io as sio
from os import path

import argparse

DATASET = 'Dataset/'

def parseZic(filename):
    print "Parsing : ",filename
    song = ms.converter.parse(DATASET+filename)

    numP = 0
    for p in song:
        if numP>0:
            print "Remove", p
            song.remove(p)
        else:
            numP += 1
            numV = 0
            for v in p:
                print v
                if type(v) is ms.stream.Voice:
                    print "isVoice ",numV
                    if numV>0:
                        print "Deleting", numV
                        p.remove(v)
                    else:
                        for n in v:
                            if type(n) is ms.chord.Chord:
                                note = ms.note.Note(n.root())
                                note.duration = n.duration
                                v.replace(n,note)
                                
                        numV += 1
                    

    # song.show('text')
    song = song.flat
    #song.show()

    saveMusicToMat(song, 'paco')

def parseAllDataset():

    for filename in os.listdir(DATASET):
        if filename.endswith(".mid"):
            parseZic(filename)
            break
        else:
            continue


def createDummySet1():
    s = ms.stream.Stream()
    n1 = ms.note.Note()
    n1.pitch.name = 'E3'
    n1.duration.type = 'quarter'
    n1.duration.quarterLength

    n2 = ms.note.Note()
    n2.pitch.name = 'A3'
    n2.duration.type = 'quarter'
    n2.duration.quarterLength

    for m in range(20):
        s.repeatAppend(n1,7)
        s.repeatAppend(n2,1)

    s.show('text')

    saveMusicToMat(s)


def createDummySet2():
    s = ms.stream.Stream()
    n1 = ms.note.Note()
    n1.pitch.name = 'E3'
    n1.duration.quarterLength=0.125

    n2 = ms.note.Note()
    n2.pitch.name = 'A3'
    n2.duration.quarterLength=0.125

    n3 = ms.note.Note()
    n3.pitch.name = 'C3'
    n3.duration.quarterLength=0.125

    n4 = ms.note.Note()
    n4.pitch.name = 'C4'
    n4.duration.quarterLength=0.125


    for m in range(100):
        s.repeatAppend(n1,1)
        s.repeatAppend(n2,1)
        s.repeatAppend(n3,1)
        s.repeatAppend(n2,1)
        s.repeatAppend(n4,1)


    s.show()

    saveMusicToMat(s,'dummy')

    
def saveMusicToMat(seq,fileName='dummy'):

    #So a note is a vector of 128 features, onehot if single note, multiple if chord
    dictNote = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
    dictNote['E-'] = 3
    dictNote['B-'] = 10

    numNotes = len(seq)
    numFeatures = 128
    
    partition = np.zeros((numNotes,numFeatures))

    position = 0
    for i,n in enumerate(seq):
        if type(n) is ms.note.Note:
            duration = int(np.round(n.duration.quarterLength*32/4))
            index = n.octave*12 + dictNote[n.name]
            partition[position:position+duration,index] = 1

            position += duration

    sio.savemat(fileName,{'mat':partition})

def parseSeq(fileName):
    listNote = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    s = ms.stream.Stream()

    if fileName[-4:] == '.mat':
        seq = sio.loadmat(fileName)['x']
    else:
        seq = sio.loadmat(fileName+'.mat')['x']


    elem_1 = 'lol'
    numTimes = 1
    
    for i,elem in enumerate(seq):

        n = ms.note.Note()
        octave, note = divmod(elem[0],12)
        n.pitch.name = listNote[int(note)-1]+str(int(octave))
        print("pitch.name",n.pitch.name)
        
        if n.pitch.name==elem_1 and i%32!=0:
            numTimes += 1
        else:
            n.duration.quarterLength = 0.125*numTimes

            elem_1 = n.pitch.name
            numTimes = 1
            print("elem-1 : APPEND",elem_1)
            print("n.duration",n.duration.quarterLength)
            s.append(n)

    s.show()

    
parser = argparse.ArgumentParser()
parser.add_argument('-c',"--create",help="type of dataset you want to create",default='')
parser.add_argument('-r',"--read", help="name of music seq you want to parse",default='')

args = parser.parse_args()
print args,'\n'

if args.create:
    if args.create=='dummy1':
        createDummySet1()
    elif args.create=='dummy2':
        createDummySet2()
    elif args.create=='flamenco':
        parseZic('Lucia_Rio_Ancho.mid')
    elif args.create=='flamenco2':
        parseZic('Paco_de_Lucia_Entre_dos_Aguas.mid')
    else:
        raise NotImplemented("Fuck you")
elif args.read:
    parseSeq(args.read)
else:
    print("Need argument --read or -- create")

#!/usr/bin/python
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine, euclidean
import pickle
import gc
import codecs
import sys

def getlines():
    f1 = codecs.open("w2vec100k.txt",'r',"utf-8")

    while f1:
        l1 = f1.readline()
        w1 = "".join(l1.strip().split()[:-200])

        v1 = np.array(map(float,l1.strip().split()[-200:]))

        v1 = np.array(map(float,l1.strip().split()[-200:]))

        yield w1,v1

def alllines():
    d1 = {}
    ctr = 0
    for w,v1 in getlines():
        ctr += 1
        if ctr % 10000 == 0:
            print ctr
        if ctr >= 50000: break
        d1[w] = v1
    return d1

d1 = alllines()

minD = None
maxD = None
avgD = 0.0

minV = []
maxV = []
avgV = []
totV = 0

tot = 0

for w1 in d1:
    minV.append(None)
    maxV.append(None)
    avgV.append(0.0)
    totV = 0
    for w2 in d1:
        if w1 == w2:
            continue
        #d = euclidean(d1[w1],d1[w2])
        d = cosine(d1[w1],d1[w2])
        avgD += d
        tot += 1

        if minD is None or minD > d:
            minD = d

        if maxD is None or maxD < d:
            maxD = d

        if minV[-1] is None or minV[-1] > d:
            minV[-1] = d
        if maxV[-1] is None or maxV[-1] < d:
            maxV[-1] = d

        avgV[-1] += d
        totV += 1
    
    avgV[-1] /= totV
    if len(maxV) % 100 == 0: print tot

print minD, avgD/tot, maxD

def avg(l):
    return sum(l)/len(l)

print avg(minV), avg(avgV), avg(maxV)

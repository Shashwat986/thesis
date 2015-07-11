#!/usr/bin/python
import numpy as np
from scipy.spatial.distance import cosine, euclidean
import pickle
import codecs
import sys

#_f1 = 'vec100k1.txt'
_f1 = 'w2vec100k.txt'
#_f2 = 'vec100k2.txt'
_f2 = 'w2vec100k2.txt'
#_n1 = 'nn100k_15_1.txt'
_n1 = 'w2nn100k_15.txt'
#_n2 = 'nn100k_15_2.txt'
_n2 = 'w2nn100k_152.txt'

if len(sys.argv) > 1:
    k = int(sys.argv[1])
else:
    k = 6

def getlines():
    f1 = codecs.open(_f1,'r',"utf-8")
    f2 = codecs.open(_f2,'r',"utf-8")

    d1 = {}
    d2 = {}
    l1 = " "
    l2 = " "    

    while l1 and l2:
        l1 = f1.readline()
        l2 = f2.readline()
        w1 = "".join(l1.strip().split()[:-200])
        w2 = "".join(l2.strip().split()[:-200])

        v1 = np.array(map(float,l1.strip().split()[-200:]))
        v2 = np.array(map(float,l2.strip().split()[-200:]))
        
        if v1.shape != v2.shape or len(v1) == 0:
            continue
        d1[w1] = v1
        d2[w2] = v2
        
    return (d1,d2)

def getnns():
    f1 = codecs.open(_n1,'r',"utf-8")
    f2 = codecs.open(_n2,'r',"utf-8")
    l1 = " "
    l2 = " "    
    d1 = {}
    d2 = {}

    while l1 and l2:
        l1 = f1.readline()
        l2 = f2.readline()

        w1 = "".join(l1.split('\t')[:-1])
        w2 = "".join(l2.split('\t')[:-1])

        n1 = l1.strip().split('\t')[-1].split(',')
        n1 = [w.strip().split(' ')[0] for w in n1]
        n2 = l2.strip().split('\t')[-1].split(',')
        n2 = [w.strip().split(' ')[0] for w in n2]
        d1[w1] = n1[:k]
        d2[w2] = n2[:k]
    return d1,d2

print "Reading files"
d1,d2 = getlines()
nn1,nn2 = getnns()
print "Done"

def generator_nns(nn1,nn2):
    for w in nn1:
        if w in nn2:
            yield w,nn1[w],w,nn2[w]

gen = generator_nns(nn1,nn2)

# Testing
diff_euc = 0.0
diff_cos = 0.0
tot = 200
ctr = 0
for n in xrange(tot):
    w1,n1,w2,n2 = gen.next()
    if n % 20 == 0: print n
    
    if w2 not in d2:
        print "Error"
        tot -= 1
        continue
    v1 = d1[w1]
    v2 = d2[w2]
    v2p = np.zeros((200,),dtype=np.float64)
    ctr_v = 0
    for word in n1:
        if word in d2:
            v2p += d2[word] #* euclidean(d1[word],v1)
            ctr_v += 1#euclidean(d1[word],v1)
    v2p /= ctr_v
    
    diff_euc += euclidean(v2p, v2.reshape(200,1))
    diff_cos += cosine(v2p,v2.reshape(200,1))

    '''
    NNS = 5
    if True:
        # Algo to calculate nearest neighbours of the new vector (v2p)
        min5 = []
        for word in d2:
            dist = euclidean(d2[word],v2p)
            if len(min5) < NNS:
                min5.append((dist,word))
            if dist < max(min5)[0]:
                min5.append((dist,word))
                min5.sort()
                min5 = min5[:NNS]
        # Algo ends
        ctr += 1 if len(set(n2[:NNS]) & set([m[1] for m in min5])) else 0
    ''' 


diff_euc /= tot
diff_cos /= tot

print diff_euc
print diff_cos

print "ACC"
print ctr,'/',tot,'=',1.0*ctr/(tot)

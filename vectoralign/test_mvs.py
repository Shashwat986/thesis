#!/usr/bin/python
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine, euclidean
import pickle
import gc
import codecs
import sys

#_f1 = 'vec100k1.txt'
_f1 = 'w2vec100k.txt'
_f2 = 'w2vec100k2.txt'
#_f2 = 'vec100k2.txt'
#_n1 = 'nn100k_15_1.txt'
_n1 = 'w2nn100k_15.txt'
_n2 = 'w2nn100k_152.txt'
#_n2 = 'nn100k_15_2.txt'

order = []

def alllines():
    f1 = codecs.open(_f1,'r',"utf-8")
    f2 = codecs.open(_f2,'r',"utf-8")
    l1 = " "
    l2 = " "

    d1 = {}
    d2 = {}

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

        if w1 not in order: order.append(w1)
        if w2 not in order: order.append(w2)

        if len(d1) % 10000 == 0: print len(d1)
    return d1,d2
    
def getlines():
    for w in order:
        if w in d2 and w in d1:
            yield w,d1[w],d2[w]

d1,d2 = alllines()

def getnns():
    d1 = {}
    d2 = {}
    f1 = codecs.open(_n1,'r',"utf-8")
    f2 = codecs.open(_n2,'r',"utf-8")
    l1 = " "
    l2 = " "    

    while l1 and l2:
        l1 = f1.readline()
        l2 = f2.readline()

        w1 = "".join(l1.split('\t')[:-1])
        w2 = "".join(l2.split('\t')[:-1])

        n1 = l1.strip().split('\t')[-1].split(',')
        n1 = [w.strip().split(' ')[0] for w in n1]
        n2 = l2.strip().split('\t')[-1].split(',')
        n2 = [w.strip().split(' ')[0] for w in n2]

        d1[w1] = n1
        d2[w2] = n2
    return d1,d2

nn1,nn2 = getnns()

if len(sys.argv) > 1:
    savefile = sys.argv[1]
else:
    savefile = None


# Y = A.X
d = 200
n_extra = 800
Y = np.zeros((d + 1, d + 1 + n_extra))
X = np.zeros((d + 1, d + 1 + n_extra))

print "Creating matrix"
gen = getlines()
ctr = -1
for n in xrange(201 + n_extra):
    w,v1,v2 = gen.next()
    Y[:,n] = np.append(v2,1)
    X[:,n] = np.append(v1,1)

if savefile:
    print "Matrix loaded from file"
    with open(savefile,'rb') as sfp:
        Ab = pickle.load(sfp)
else:
    print "Matrix created. Solving"
    gc.collect()
    print X
    print Y
    Ab = np.dot(Y,np.linalg.pinv(X))
    print np.allclose(np.dot(Ab,X), Y) 
    print "Solved. Saving"
    with open("matr_m.dat","wb") as sfp:
        pickle.dump(Ab,sfp)
    if np.any(Ab):
        print "Yes!"
    else:
        print "No!"

print Ab.shape
print np.allclose(np.dot(Ab,X), Y) 

# v2 = Ab . v1

# Testing
diff_cos = 0.0
diff_euc = 0.0
tot = 200
ctr = 0

# Discarding these words
#'''
for _ in xrange(1000 - n_extra):
	_ = gen.next()
#'''

try:
    NNS = int(sys.argv[2])
except:
    NNS = 15

for n in xrange(tot):
    if n % 20 == 0: print n
    w,v1,v2 = gen.next()
    v2p = np.delete(np.dot(Ab,np.append(v1,1)).reshape((201,1)),-1)

    diff_cos += cosine(v2p, v2.reshape((200,1)))
    diff_euc += euclidean(v2p, v2.reshape((200,1)))

    if False:#True:
        min5 = []
        for word in d2:
            if len(d2[word]) != len(v2p):
                continue
            dist = euclidean(d2[word],v2p)
            if len(min5) < NNS:
                min5.append((dist,word))
            if dist < max(min5)[0]:
                min5.append((dist,word))
                min5.sort()
                min5 = min5[:NNS]
        #print set(nn2.get(w,[])[:NNS]), set([m[1] for m in min5])
        ctr += 1 if w in [m[1] for m in min5] else 0
        print w, min5
        #len(set(nn2.get(w,[]))&set([m[1] for m in min5]))


diff_euc /= tot
diff_cos /= tot

print diff_euc
print diff_cos

print "ACC"
print ctr ,'/',tot,'=',1.0*ctr/(tot)

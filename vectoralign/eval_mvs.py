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

def find_nn(v,d, num = 1):
    minv = []
    for word in d:
        if d[word].shape != v.shape: continue
        dist = cosine(d[word],v)
        if len(minv) < num:
            minv.append((dist,word))
        if dist < min(minv)[0]:
            minv.append((dist,word))
            minv.sort()
            minv = minv[:num]
    if num == 1:
        return minv[0]
    else:
        return minv

def get_nn_vec(w):
    ctr_v = 0
    v2p = np.zeros((200,),dtype=np.float64)
    for word in nn2[w]:
        if word in d2:
            v2p += d2[word]
            ctr_v += 1
    v2p /= ctr_v
    return v2p

def get_new_vec(w):
    v1 = d1[w]
    v2p = np.delete(np.dot(Ab,np.append(v1,1)).reshape((201,1)),-1)
    return v2p.reshape((200,))

# Testing
tot = 0
ctr = 0
ctrnn = 0
ctrgs = 0
ctrgs2 = 0
fp = open("questions-words.txt","r")
type_ctr = 0
ctr_in = 0
for l in fp:
    if len(l) == 0  or l[0] == ":":
        type_ctr += 1
        ctr_in = 0
        print "FINAL SCORE: ", ctr,ctrnn,ctrgs,ctrgs2,tot
        #ctr,ctrgs,ctrgs2 = 0,0,0
        tot = 0
        print l.strip()
        continue
    w = l.strip().split()
    if type_ctr not in [1,5,7,9,12]: continue
    #if type_ctr not in [1,5]: continue
    #if type_ctr not in [9,12,7]: continue
    if len(w) != 4 or any([i not in d2 for i in w]):
        continue

    ctr_in += 1
    if ctr_in > 50: continue

    tot += 1
    if tot % 10 == 0: print ctr,ctrgs,tot
    v1 = d1[w[2]] - d1[w[0]] + d1[w[1]]
    v2 = d2[w[2]] - d2[w[0]] + d2[w[1]]
    v2p = get_new_vec(w[2]) - get_new_vec(w[0]) + get_new_vec(w[1])
    v2n = get_nn_vec(w[2]) - get_nn_vec(w[0]) + get_nn_vec(w[1])

    if find_nn(v1,d1)[1] == w[3]:
        ctrgs2 += 1
    if find_nn(v2,d2)[1] == w[3]:
        ctrgs += 1
    if find_nn(v2p,d2)[1] == w[3]:
        ctr += 1
    if find_nn(v2n,d2)[1] == w[3]:
        ctrnn += 1
 
print "ACC"
print ctr,'/',tot,'=',1.0*ctr/(tot)
print ctrnn,'/',tot,'=',1.0*ctrnn/(tot)
print "GSACC"
print ctrgs,'/',tot,'=',1.0*ctrgs/(tot)
print ctrgs2,'/',tot,'=',1.0*ctrgs2/(tot)


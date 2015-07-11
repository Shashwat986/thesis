#!/usr/bin/python
import numpy as np
from scipy.spatial.distance import cosine, euclidean
import pickle
import codecs
import sys

_f1 = 'w2vec100k.txt'
#_f1 = 'vec100k1.txt'
_f2 = 'w2vec100k2.txt'
#_f2 = 'vec100k2.txt'
_n1 = 'w2nn100k_15.txt'
#_n1 = 'nn100k_15_1.txt'
_n2 = 'w2nn100k_152.txt'
#_n2 = 'nn100k_15_2.txt'

if len(sys.argv) > 1:
    k = int(sys.argv[1])
else:
    k = 6

order = []
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

        if w1 not in order: order.append(w1)
        if w2 not in order: order.append(w2)

        
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

def find_nn(v,d, num = 1):
    minv = []
    for word in d:
        dist = cosine(d[word],v)
        if len(minv) < num:
            minv.append((dist,word))
        if dist < min(minv)[0]:
            minv.append((dist,word))
            minv.sort()
            min5 = min5[:num]
    if num == 1:
        return minv
    else:
        return minv

def get_new_vec(w):
    ctr_v = 0
    v2p = np.zeros((200,),dtype=np.float64)
    for word in nn2[w]:
        if word in d2:
            v2p += d2[word]
            ctr_v += 1
    v2p /= ctr_v
    return v2p

# Testing
tot = 0
ctr = 0
ctrgs = 0
ctrgs2 = 0
fp = open("questions-words.txt","r")
type_ctr = 0
ctr_in = 0
for l in fp:
    if len(l) == 0 or l[0] == ':':
        type_ctr += 1
        ctr_in = 0
        print "FINAL SCORE: ",ctr,ctrgs,ctrgs2,tot
        ctr = 0
        ctrgs = 0
        ctrgs2 = 0
        tot = 0
        print type_ctr,l.strip()
        continue
    w = l.strip().split()
    #if type_ctr not in [3]: continue
    #if type_ctr not in [1,5]: continue
    #if type_ctr not in [9,12,7]: continue
    if len(w) != 4 or any([i not in d2 for i in w]):
        continue
    
    ctr_in += 1
    if ctr_in > 50: continue

    tot += 1
    if tot % 10 == 0: print tot
    v2 = d2[w[2]] - d2[w[0]] + d2[w[1]]
    v1 = d1[w[2]] - d1[w[0]] + d1[w[1]]
    #v2p = d2[w[2]] - d2[w[0]] + get_new_vec(w[1])
    v2p = get_new_vec(w[2]) - get_new_vec(w[0]) + get_new_vec(w[1])
    ''' 
    nn_closest = 17.9
    if euclidean(v1,d1[w[3]]) < nn_closest:
        ctrgs2 += 1
    if euclidean(v2,d2[w[3]]) < nn_closest:
        ctrgs += 1
    if euclidean(v2p, get_new_vec(w[3])) < nn_closest:
        ctr += 1
    '''
    if find_nn(v1,d1)[1] == w[3]:
        ctrgs2 += 1
    if find_nn(v2,d2)[1] == w[3]:
        ctrgs += 1
    if find_nn(v2p,d2)[1] == w[3]:
        ctr += 1
    #'''

print "ACC"
print ctr,'/',tot,'=',1.0*ctr/(tot)
print "GSACC"
print ctrgs,'/',tot,'=',1.0*ctrgs/(tot)
print ctrgs2,'/',tot,'=',1.0*ctrgs2/(tot)

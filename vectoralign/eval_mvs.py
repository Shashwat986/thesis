#!/usr/bin/python
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine, euclidean
import pickle
import gc
import codecs
import sys

''' This program requires 4 files as input. _f1, _f2, _n1, and _n2.
_f1, _f2 : These files contain the word vectors for words in their respective corpora.
_n1, _n2 : These files contain the nearest neighbours of each word in the respective corpora,
           based on the vectors in _f1 and _f2. We recommend having the 15 nearest neighbours,
           in this file. Note that the file `postprocessing.py` generates these nearest
           neighbours in the format understood by this program, given the word vector file.
'''

#_f1 = 'vec100k1.txt'
_f1 = 'w2vec100k.txt'
_f2 = 'w2vec100k2.txt'
#_f2 = 'vec100k2.txt'
#_n1 = 'nn100k_15_1.txt'
_n1 = 'w2nn100k_15.txt'
_n2 = 'w2nn100k_152.txt'
#_n2 = 'nn100k_15_2.txt'

d = 200

''' This list is used to keep track of word occourrences.
This is because the order of keys in the dictionary is not preserved, while the words in the input files are
sorted in the order of decreasing frequency. This variable keeps an approximate ordering based on frequency
'''
order = []

def alllines():
    '''This function returns two dictionaries.
    These dictionaries correspond to the word vectors in _f1 and _f2 respectively.
    Each key in a dictionary corresponds to a word present in the respective file.
    Each value represents a numpy array with the vector of the word.
    '''
    f1 = codecs.open(_f1,'r',"utf-8")
    f2 = codecs.open(_f2,'r',"utf-8")
    l1 = " "
    l2 = " "

    d1 = {}
    d2 = {}

    while l1 and l2:
        l1 = f1.readline()
        l2 = f2.readline() 
        w1 = "".join(l1.strip().split()[:-d])
        w2 = "".join(l2.strip().split()[:-d])

        v1 = np.array(map(float,l1.strip().split()[-d:]))
        v2 = np.array(map(float,l2.strip().split()[-d:]))
        
        if v1.shape != v2.shape or len(v1) == 0:
            continue
        d1[w1] = v1
        d2[w2] = v2

        if w1 not in order: order.append(w1)
        if w2 not in order: order.append(w2)

        if len(d1) % 10000 == 0: print len(d1)
    return d1,d2
    
def getlines():
    '''This is a generator for the words in the dictionaries.
    This will only return words in the vocabularies of both corpora.
    '''
    for w in order:
        if w in d2 and w in d1:
            yield w,d1[w],d2[w]

# We calculate and store the dictionaries generated using `alllines()`
d1,d2 = alllines()


def getnns():
    ''' This function returns two dictionaries.
    Each dictionary maps words in the respective corpus to a list of nearest neighbours.
    '''
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

# We calculate and store the dictionaries generated using `getnns()`
nn1,nn2 = getnns()

# Checks if we are loading from a file or not. If we are, this file will be loaded later.
if len(sys.argv) > 1:
    savefile = sys.argv[1]
else:
    savefile = None

#d = 200

# The number of extra points we want matched.
n_extra = 800

# Y = A.X
Y = np.zeros((d + 1, d + 1 + n_extra))
X = np.zeros((d + 1, d + 1 + n_extra))

print "Creating matrix"
gen = getlines()
ctr = -1
for n in xrange(d + 1 + n_extra):
    w,v1,v2 = gen.next()
    # Augmenting 1 so that we can deal with the bias vector `b`.
    Y[:,n] = np.append(v2,1)
    X[:,n] = np.append(v1,1)

if savefile:
    # Loading matrix solutions from the savefile
    print "Matrix loaded from file"
    with open(savefile,'rb') as sfp:
        Ab = pickle.load(sfp)
else:
    # Solving the matrix.
    print "Matrix created. Solving"
    gc.collect()
    print X
    print Y
    Ab = np.dot(Y,np.linalg.pinv(X))
    # Sanity-check
    print np.allclose(np.dot(Ab,X), Y)
    
    # This newly-solved matrix is saved by default. This file is overwritten every time we re-solve the matrix.
    print "Solved. Saving"
    with open("matr_m.dat","wb") as sfp:
        pickle.dump(Ab,sfp)
    
    # Sanity-check
    if np.any(Ab):
        print "Yes!"
    else:
        print "No!"

print Ab.shape
print np.allclose(np.dot(Ab,X), Y) 

# v2 = Ab . v1

def find_nn(v,d, num = 1):
    ''' This function find the (num) nearest neighbours of a vector v,
    in the set of vectors represented by the dictionary d.
    Input: v (numpy array), d (dictionary: key (word) => value (word vector)), num (number of NNs).
    Output: List of words of length num (if num > 1). One word (if num == 1).
    '''
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
    ''' This function calculates and returns the predicted vector using the global equivalence approach.
    '''
    ctr_v = 0
    v2p = np.zeros((d,),dtype=np.float64)
    for word in nn2[w]:
        if word in d2:
            v2p += d2[word]
            ctr_v += 1
    v2p /= ctr_v
    return v2p

def get_new_vec(w):
    ''' This function calculates and returns the predicted vector using the local equivalence approach.
    '''
    v1 = d1[w]
    v2p = np.delete(np.dot(Ab,np.append(v1,1)).reshape((d + 1,1)),-1)
    return v2p.reshape((d,))

# Testing
tot = 0
ctr = 0
ctrnn = 0
ctrgs = 0
ctrgs2 = 0

# Dataset from Word2Vec.
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
    
    # Testing
    if type_ctr not in [1,5,7,9,12]: continue
    #if type_ctr not in [1,5]: continue
    #if type_ctr not in [9,12,7]: continue
    if len(w) != 4 or any([i not in d2 for i in w]):
        continue
    
    # Testing
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


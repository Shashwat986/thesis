# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster import hierarchy as hier
from scipy.spatial import distance
import json
import codecs
import sys

if len(sys.argv) < 2:
	print "Provide file name"
	sys.exit(1)
elif len(sys.argv) < 3:
	out_file = "nn9m.dat"
else:
	out_file = sys.argv[2]

print "Start"
fi = codecs.open(sys.argv[1],"r","utf-8")
words = []
data = []
for line in fi:
	
	if not len(line.strip()): continue
	k = line.strip().split()
	words.append(k[0])
	data.append([float(i) for i in k[-200:]])

fi.close()

vectors = np.array(data)

print "Pre-processing done"

# Calculate the distance matrix
def dist(x,y):
	return np.dot(x,y)

knn = KNeighborsClassifier()
knn.fit(vectors,[0]*len(vectors))

fo = codecs.open(out_file,"w","utf-8")
for i,word in enumerate(words):
	d,n = knn.kneighbors(vectors[i], n_neighbors = 25, return_distance = True)
	if i%1000==0: print d,n
	fo.write(word+"\t")
	for j in range(1,len(n[0])):
		fo.write(words[n[0][j]]+" ({:.6f}), ".format(d[0][j]))
	fo.write("\n")
fo.close()




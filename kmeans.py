#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description

Usage:
    $python kmeans.py -f DATASET.csv -k No.clusters

    $python kmeans.py -f crater.csv -k 3
"""
from sklearn.cluster import KMeans
import csv
from collections import defaultdict
from optparse import OptionParser
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd

##################
# クラスタリング結果を返すように実装してください


def compute_dis(vecA, vecB):

    distance = np.sqrt(sum(np.power(vecA-vecB, 2)))
    return distance

def rand_cent(data, k):

    n = data.shape[1]
    centros = np.zeros((k,n))
    for j in range(n):
        minJ = min(data[:,j])
        maxJ = max(data[:,j])
        rangej = float(maxJ - minJ)
        centros[:,j] = minJ + rangej*np.random.rand(k)
    return centros

def clustering(data, k):
    m = data.shape[0]

    clusterAssign = np.zeros((m,2))
    centros = rand_cent(data, k)
    cluster_changed = True
    while cluster_changed:
        for i in range(m):
            minDist = float('inf')
            minIndex = -1
            for j in range(k):
                dist = compute_dis(data[i,:], centros[j,:])
                if (dist < minDist):
                    minDist = dist
                    minIndex = j
            clusterAssign[i, :] = minIndex, minDist**2

        last_centros = centros.copy()
        print(centros)

        for cent in range(k):
            target_set = []
            for i in range(m):
                if (clusterAssign[i,0] == cent):
                    target_set.append(data[i,:])
            centros[cent,:] = np.mean(target_set)

        if (np.array_equal(last_centros,centros)):
            cluster_changed = False
    return clusterAssign[:,0]

##################

def dataFromFile(fname):
        """Function which reads from the file and yields a generator"""
        file_iter = open(fname, 'rU')
        for line in file_iter:
                line = line.strip().rstrip(',')                         # Remove trailing comma
                record = line.split(',')
                yield record
if __name__ == '__main__':

    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv',
                         default=None)
    optparser.add_option('-k',
                         dest='k',
                         help='number of clusters',
                         default=3,
                         type='int')
    (options, args) = optparser.parse_args()
    inFile = None
    if options.input is None:
            data = pd.read_csv('crater.csv')
    elif options.input is not None:
#            inFile = dataFromFile(options.input)
            data = pd.read_csv(options.input)
    else:
            print('No dataset filename specified, system with exit\n')
            sys.exit('System will exit')
    k = options.k
##################
#    pred は以下のようなリストが期待されます
#    [1,0,0,2,1,0]
#    この場合、要素の一つ目がクラスタ1に、二つ目がクラスタ0に属していることを意味しています
##################
#    feature=[]
#    for record in inFile:
#        feature.append(list(map(float,record)))
    feature = data.values
    pred = clustering(feature,k)

#plot nodes
    plt.title("kmeans")
    x=[]
    y=[]
    for i in range(len(set(pred))):
        x.append([])
        y.append([])
    for i in range(len(pred)):
        x[int(pred[i])].append(feature[i,0])
        y[int(pred[i])].append(feature[i,1])
    color_list=['red','blue','yellow','green','purple','c', 'olivedrab']
    for i in range(len(x)):
        plt.scatter(x[i],y[i],label=i, c=color_list[i])
    plt.legend()
    plt.show()

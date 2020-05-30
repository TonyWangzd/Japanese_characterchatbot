#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description

Usage:
    $python dbscan.py -f DATASET.csv -e EPSILON -m minPoitns

    $python dbscan.py -f crater.csv -e 0.8 -m 2
"""
from sklearn.cluster import DBSCAN
import csv
from collections import defaultdict
from optparse import OptionParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

##################
# クラスタリング結果を返すように実装してください
# two column of matrix disatance_list, one is the distance of (i,j), one is if it smaller than epsilon
def clustering(data, eps, minPoints):

    m = data.shape[0]
    label_list = np.zeros(m) # 0 for don't have label, -1 as noise, 1, 2, 3 as labels
    label_number = 1        # start the label from 1

    def compute_neighbor(p):  # the point in range epsilon called neighbor
        NN =[]
        for i in range(m):
            dist = compute_dis(data[p,:], data[i,:])
            if dist < eps:
                NN.append(i)
        return NN

    for p in range(m):

        if (label_list[p] == 0):

            NN = compute_neighbor(p)
            if len(NN) >= minPoints:
                cluster_set = []
                for j in NN:
                    cluster_set.append(j)

                label_list[p] = label_number # give the point an number as label, visited

                for point in cluster_set:
                    if (label_list[point] == 0):

                        NN = compute_neighbor(point)
                        if len(NN) >= minPoints:
                            for j in NN:
                                cluster_set.append(j)
                            cluster_set = list(set(cluster_set))
                        label_list[point] = label_number # give the point an number as label, visited

                label_number += 1    # change the current lable

            else:
                label_list[p] = -1   # consider the point as noise


    return label_list

def compute_dis(vecA, vecB):

    distance = np.sqrt(sum(np.power(vecA-vecB, 2)))
    return distance
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
    optparser.add_option('-e', '--epsilon',
                         dest='eps',
                         help='threshold for marge step',
                         default=0.8,
                         type='float')
    optparser.add_option('-m', '--minPoints',
                         dest='minPoints',
                         help='minimum number of points for cluster',
                         default=20,
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

    eps = options.eps
    minPoints = options.minPoints
    ##################
    #    pred は以下のようなリストが期待されます
    #    [1,0,0,2,1,0]
    #    この場合、要素の一つ目がクラスタ1に、二つ目がクラスタ0に属していることを意味しています
    ##################
    #    feature=[]
    #    for record in inFile:
    #        feature.append(list(map(float,record)))
    feature = data.values
    pred = clustering(feature, eps, minPoints)
    # plot nodes
    plt.title("DBSCAN")
    x = []
    y = []
    for i in range(len(set(pred))+1):
        x.append([])
        y.append([])
    for i in range(len(pred)):
        x[int(pred[i])].append(feature[i, 0])
        y[int(pred[i])].append(feature[i, 1])
    color_list = ['red', 'blue', 'yellow', 'green', 'purple', 'c', 'olivedrab']
    for i in range(len(x)):
        plt.scatter(x[i], y[i], label=i, c=color_list[i])
    plt.legend()
    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 11:59:54 2020

@author: likem
"""


import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch


def read_trajs():
    total_length = []
    totoal_time = []
    with h5py.File('F/data/beijing.h5', 'r') as f:
    
        for ii in tqdm(range(f.attrs['num'])):
            trip = np.array(f.get('trips/'+str(ii+1)))
            ts = np.array(f.get('timestamps/'+str(ii+1)))
            total_length.append(len(ts))
            totoal_time.append(ts[-1]-ts[0])
        print(sum(total_length)/f.attrs['num'])
        print(sum(totoal_time)/f.attrs['num'])
        plt.figure(1)
        plt.plot(total_length,'r*')
        plt.figure(2)
        plt.plot(totoal_time,'r*')


def common_values(m, n):
    len_m = len(m)
    len_n = len(n)
    i, j = 0, 0
    common_value = []
    while i < len_m and j < len_n:
        if m[i] == n[j]:
            common_value.append(m[i])
            i += 1
            j += 1
        elif m[i] < n[j]:
            i += 1
        else:
            j += 1
    return common_value





if __name__ == "__main__":
    # i = 100
    # with h5py.File("E:/data/porto.h5", 'r') as f:
    #     print(f.attrs['num'])
    #     trip = np.array(f.get('trips/'+str(i+1)))
    #     ts = np.array(f.get('timestamps/'+str(i+1)))
    # print(trip)
    # print(ts)

    # from tqdm import tqdm
    # a, b = [], []
    # for ii in range(1000000):
    #     a.append(sorted(np.random.randint(1,10000,40).tolist()))
    #     b.append(sorted(np.random.randint(1, 10000, 40).tolist()))
    # import time
    # t1 = time.time()
    # for item1, item2 in zip(a,b):
    #     cv = common_values(item1, item2)
    # print(time.time()-t1)
    # t1 = time.time()
    # for item1, item2 in zip(a,b):
    #     cv = list(set(item1).intersection(set(item2)))
    # print(time.time() - t1)

    n = 10000
    x = torch.tensor(np.random.randn(n,10).tolist())
    dist_matrix = euclidean_dist(x, x)
    print(dist_matrix)

    
    
    
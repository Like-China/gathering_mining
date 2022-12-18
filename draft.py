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


if __name__ == "__main__":
    # i = 100
    # with h5py.File("E:/data/porto.h5", 'r') as f:
    #     print(f.attrs['num'])
    #     trip = np.array(f.get('trips/'+str(i+1)))
    #     ts = np.array(f.get('timestamps/'+str(i+1)))
    # print(trip)
    # print(ts)


    n = 10000
    x = torch.tensor(np.random.randn(n,10).tolist())
    dist_matrix = euclidean_dist(x, x)
    print(dist_matrix)

    
    
    
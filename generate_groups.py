# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 19:02:31 2021

@author: likem
"""
import numpy as np
import random
from tqdm import tqdm

'''
xyt转换为时空id
'''
def xyt2id(x, y, t, x_size, y_size):
    return t*(x_size*y_size)+y*x_size+x

''' 
创建模拟轨迹
最大轨迹编号为x_size*y_size*t_size-1
'''
def create_trj(x_size, y_size, t_size,trj_num,trj_len):
    all_trjs = []
    for i in  range(trj_num):
        trj = []
        x = np.random.randint(x_size)
        y = np.random.randint(y_size)
        t = np.random.randint(t_size-trj_len)
        cellId = xyt2id(x, y, t, x_size, y_size)
        trj.append(cellId)
        for j in range(trj_len-1):
            if (x == x_size-1):
                x = random.choice([x-1,x])
            elif (x == 0):
                x = random.choice([x+1,x])
            else:
                x = random.choice([x+1,x, x-1])
                
            if (y == y_size-1):
                y = random.choice([y-1,y])
            elif (y == 0):
                y = random.choice([y+1,y])
            else:
                y = random.choice([y+1,y, y-1])
            
            t += 1
            cellId = xyt2id(x, y, t, x_size, y_size)
            if(cellId>=t_size*x_size*y_size):
                print(cellId)
            trj.append(cellId)
        all_trjs.append(trj)
    return all_trjs



'''
求两个有序序列的公共序列
'''
def commom_vals(m, n):
    minLen = min(len(m),len(n))
    if(m[0]>=n[-1] or n[0]>=m[-1]):
        return []
    i = 0
    j = 0
    cvals = []
    while (j < minLen and i < minLen):
        if (m[i] == n[j]):
            cvals.append(m[i]);
            i += 1
            j += 1
        elif (m[i] < n[j]):
            i += 1
        else:
            j += 1
    return cvals;



'''
输出每条轨迹属于哪个组的映射，列表形式
若不属于某个组，则赋值为-1
'''
def trj2groupId(all_groups,trj_num):
    gIDs = [-1]*trj_num
    for ii in range(len(all_groups)):
        for trj_id in all_groups[ii]:
            gIDs[trj_id] = ii
    return gIDs
    


''' 
输出全部的group
'''
def get_groups(trjs, min_lifetime, min_group_trj_nums):
    all_common_pairs = []
    ''' 
    记录所有的group
    '''
    all_groups = []
    all_grouped_sets = set()
    for i in tqdm(range(trj_num)):
        if i in all_grouped_sets:
            continue;
        common_pairs = []
        for j in range(i+1,trj_num):
            ''' 对已经形成group的轨迹，不再去组成新的group '''
            if j in all_grouped_sets:
                continue;
            cvals = commom_vals(trjs[i],trjs[j])
            if(len(cvals)>=min_lifetime):
                common_pairs.append([i,j,cvals])
        if(len(common_pairs)>=min_group_trj_nums):
            all_common_pairs.append(common_pairs)
            
            for ii in range(max(len(common_pairs)-min_group_trj_nums,1)):
                group = [common_pairs[ii][0],common_pairs[ii][1]]
                trj1 = common_pairs[ii][2]
                for jj in range(ii+1,len(common_pairs)):
                    trj2 = common_pairs[jj][2]
                    cvals = commom_vals(trj1,trj2)
                    if(len(cvals)>=min_lifetime):
                        group.append(common_pairs[jj][1])
                if(len(group)>=min_group_trj_nums):
                    all_groups.append(group)
                    all_grouped_sets.update(set(group))
                    break
    return all_groups
    
if __name__ == "__main__":
    '''
    生成100K只需要7s
    '''
    x_size, y_size, t_size = 10, 10, 20
    space_cell_num = x_size*y_size
    trj_num = 10000
    trj_len = 10
    trjs = create_trj(x_size, y_size, t_size,trj_num,trj_len)
    
    ''' 
    记录有共同元素的轨迹对
    返回轨迹集构成的所有 group
    '''
    min_lifetime = 2
    min_group_trj_nums = 2
    all_groups = get_groups(trjs, min_lifetime, min_group_trj_nums)
    
                
    
    
    
            

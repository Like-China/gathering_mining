import torch,os
from trainer.models import EncoderDecoder
from loader.data_utils import  pad_arrays_pair,pad_arrays_keep_invp
from settings import set_args
import numpy as np
import time
import settings as constants
from indexing.scan import  get_communities
from tqdm import tqdm




def clustering(vecs):
    t1 = time.time()
    if constants.c_method == 1:
        c = KMeans(n_clusters=constants.n).fit(vecs)
    elif constants.c_method == 2:
        c = DBSCAN(eps=constants.eps, min_samples=constants.mt).fit(vecs)
    else:
        c = AgglomerativeClustering(n_clusters=constants.n).fit(vecs)
    print('聚类用时：'+str(time.time()-t1))
    return c.labels_


# 对向量聚类后, 每个聚类分别计算group
def group_by_cluster(labels, all_trjs):
    t1 = time.time()
    all_groups1 = []
    all_group_cvs1 = []
    for i in range(constants.n):
        # 取出每个聚类中的轨迹
        cluster = np.where(labels == i)[0].tolist()
        trjs = np.array(all_trjs)[cluster].tolist()
        groups, group_cvs = get_groups(trjs, constants.min_lifetime
                                              ,constants.min_group_trj_nums)
        if len(groups)>0:
            all_groups1.append(groups)
            all_group_cvs1.append(group_cvs)
            
    if len(all_groups1) > 0:
        all_groups1 = sum(all_groups1, [])
    
    
    t2g2 = [-1]*len(all_trjs)
    for ii in range(len(all_groups1)):
        for trj_id in all_groups1[ii]:
            t2g2[trj_id] = ii
    t2 = time.time()
    print('cluster+regroup得到的组数：{}'.format(len(all_groups1)))
    print('cluster+regroup用时：{}'.format(t2-t1))
    return all_groups1, all_group_cvs1, t2g2

'''
    对每个组的共同元素进行组合
    若存在两个组共同元素大于 constants.min_lifetime
    融合这两个组为一个组
'''
def combine_and_refine(all_groups1, all_group_cvs1, t2g2):
    
    combine_pair = []
    for i in range(len(all_group_cvs1)):
        cv1 = all_group_cvs1[i]
        for j in range(i+1,len(all_group_cvs1)):
            cv2 = all_group_cvs1[j]
            if len(list(set(cv1).intersection(set(cv2))))>=constants.min_lifetime:
                combine_pair.append([i,j])
                break
    # 融合组，重新建立t2g2
    combined_G = []
    is_uncombined = [True,]*len(all_groups1)
    for (i,j) in combine_pair:
        is_uncombined[i] = False
        is_uncombined[j] = False
        g1 = all_groups1[i]
        g2 = all_groups1[j]
        g1.extend(g2)
        combined_G.append(g1)
    # 最后的组就是取未融合的组+融合的组
    uncombined_G = np.array(all_groups1)[is_uncombined].tolist()
    all_groups2 = uncombined_G.extend(combined_G)
    return combine_pair,all_groups2


def main():
    '''
        获得该批轨迹的LCS+SCAN聚类（Ground-truth)与向量距离+SCAN聚类（our mwthod)并计算召回率
    '''
    args = set_args()
    from loader.data_scaner import DataOrderScaner
    scaner = DataOrderScaner(os.path.join(args.data,"val"))
    scaner.load(10000)
    all_trjs = scaner.srcdata

    # 1. LCS+SCAN聚类
    t1 = time.time()
    connectable_trj_pair, communities, trj2community = get_groups(all_trjs)
    t2 = time.time()
    print('LCS+SCAN 分组用时：{}'.format(t2-t1))


    # 向量距离+聚类
    vecs = t2vec(args, pad_arrays_keep_invp(all_trjs))
    labels = clustering(vecs)
    get_recall(labels, all_groups) # 仅聚类后的recall
    '''3. 对每一类计算exact group 
       3.1 记录每个组的groupId
       3.2 记录每个组的common token 
       3.3 根据common token对聚类进行融合
    '''
    all_groups1, all_group_cvs1, t2g2 = group_by_cluster(labels,all_trjs)
    get_recall(t2g2, all_groups) # 仅聚类+重新分组后的recall
    combine_pair = combine_and_refine(all_groups1, all_group_cvs1, t2g2)
    return combine_pair


if __name__ == "__main__":
    
    cb = main()
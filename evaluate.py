import torch,os, h5py
from models import EncoderDecoder
from data_utils import DataOrderScaner
from t2vec import setArgs
from sklearn import metrics
import seaborn as sns
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import constants
from funcy import merge
from data_utils import pad_arrays_pair,pad_arrays_keep_invp
'''
    读取trj.t中的轨迹
    输出vecs[m0.num_layers-1]最后一层为 向量表示
'''
def t2vec(args, data):
    "read source sequences from trj.t and write the tensor into file trj.h5"
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        if torch.cuda.is_available():
            m0.cuda()
        m0.eval()
        vecs = []
        src, lengths, invp = data[0], data[1], data[2]
        if torch.cuda.is_available():
            src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
        h, _ = m0.encoder(src, lengths) # 【层数*双向2，该组轨迹个数，隐藏层数】【6，constants.n，128】
        h = m0.encoder_hn2decoder_h0(h)
        h = h.transpose(0, 1).contiguous()
        vecs.append(h[invp].cpu().data)
        
        vecs = torch.cat(vecs) # [10,3,256]
        vecs = vecs.transpose(0, 1).contiguous()  ## [3,10,256]
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
    return vecs[m0.num_layers-1].tolist()


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
    return cvals

''' 
    计算得到全部的group
'''
def get_groups(all_trjs,min_lifetime, min_group_trj_nums):
    
    all_common_pairs = []
    all_groups = []
    group_cvs = []
    trj_num = len(all_trjs)
    all_grouped_sets = set()
    for i in range(trj_num):
        if i in all_grouped_sets:
            continue;
        common_pairs = [] 
        for j in range(i+1,trj_num):
            ''' 对已经形成group的轨迹，不再去组成新的group '''
            if j in all_grouped_sets:
                continue;
            cvals = commom_vals(all_trjs[i],all_trjs[j])
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
                    group_cvs.append(cvals)
                    break
    
    return all_common_pairs



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
 
''' 
    对编码后的轨迹进行聚类
    被根据exact algorithm算法划分计算
'''
def recall_after_combine(labels, all_groups):
    
    TP_count = 0
    FN_count = 0
    for g in all_groups:
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                if labels[i] == labels[j]:
                    TP_count += 1
                else:
                    FN_count += 1
    recall = TP_count/(TP_count+FN_count)
    # print("聚类匹配数:{0},召回率：{1:.3f}".format(TP_count,recall))
    return recall
    

''' 对向量聚类后
    对每个聚类分别计算group
'''
def group_by_cluster(labels, all_trjs):
    t1 = time.time()
    all_groups1 = []
    all_group_cvs1 = []
    for i  in range(constants.n):
        # 取出每个聚类中的轨迹
        cluster = np.where(labels == i)[0].tolist()
        trjs = np.array(all_trjs)[cluster].tolist()
        groups, group_cvs = get_groups(trjs, constants.min_lifetime
                                              ,constants.min_group_trj_nums)
        if len(groups)>0:
            all_groups1.append(groups)
            all_group_cvs1.append(group_cvs)
            
    if len(all_groups1)>0:
        all_groups1 = merge(*all_groups1)
    
    
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
            if len(commom_vals(cv1, cv2))>=constants.min_lifetime:
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
    ''' 1. 获得一批轨迹
        1.1 获得该批轨迹的exact group的t2g1
        1.2 将该组轨迹进行编码
    '''
    args = setArgs()
    ## 获得数据
    scaner = DataOrderScaner(os.path.join(args.data,"val.src"))
    scaner.load(30000)
    all_trjs = scaner.srcdata
    ## 获得轨迹组成的exact group
    t1 = time.time()
    all_groups,_ = get_groups(all_trjs,constants.min_lifetime
                                              ,constants.min_group_trj_nums)
    '''
    输出每条轨迹属于哪个组的映射，列表形式
    若不属于某个组，则赋值为-1
    '''
    t2g1 = [-1]*len(all_trjs)
    for ii in range(len(all_groups)):
        for trj_id in all_groups[ii]:
            t2g1[trj_id] = ii
    
    n1 = sum(np.array(t2g1) != -1)
    n2 = sum(np.array(t2g1) == -1)
    print('归组人数:{0}\n未归组人数：{1}\n组数：{2}'.format(n1, n2,len(all_groups)))
    t2 = time.time()
    print('直接exact分组用时：{}'.format(t2-t1))
    ## 将该组轨迹进行编码
    ## 编码轨迹
    vecs = t2vec(args,pad_arrays_keep_invp(all_trjs))
    
    '''2. 对编码向量进行聚类 (constants.n类)'''
    labels = clustering(vecs)
    recall_after_combine(labels, all_groups) # 仅聚类后的recall
    '''3. 对每一类计算exact group 
       3.1 记录每个组的groupId
       3.2 记录每个组的common token 
       3.3 根据common token对聚类进行融合
    '''
    all_groups1, all_group_cvs1, t2g2 = group_by_cluster(labels,all_trjs)
    recall_after_combine(t2g2, all_groups) # 仅聚类+重新分组后的recall
    combine_pair = combine_and_refine(all_groups1, all_group_cvs1, t2g2)
    return combine_pair



    
    
    
if __name__ == "__main__":
    
    cb = main()
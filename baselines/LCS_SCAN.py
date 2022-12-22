import time
import numpy as np
from tqdm import tqdm
from indexing.scan import  get_communities
from funcy import merge
from collections import Counter

class SCAN:

    def __init__(self, min_group_trj_nums, min_lifetime):
        self.min_group_trj_nums = min_group_trj_nums
        self.min_lifetime = min_lifetime

    ''' 
        计算得到全部的group
        利用LCS+SCAN计算群组网络，返回哪些人是一个群组
        1. 记录哪些轨迹两两的LCS大于min_lifetime
        2. 连线这些轨迹并进行SCAN聚类
    '''

    def get_pairs(self, trjs):
        """
        得到每条轨迹可以和哪些轨迹形成有效的轨迹对
        :param seqs: a batch of trajectories token sequences
        :return: all of the co-movement pairs [[i,j, common_points]]
        """
        # t1 = time.time()
        all_pairs = []
        for trj in tqdm(trjs):
            pairs = []
            count = 0
            if trj.intersect_count < self.min_group_trj_nums or trj.size < self.min_lifetime:
                continue
            for ii, intersect in enumerate(trj.intersect_trjs):
                if intersect == trj.id:
                    continue
                intersect_trj = trjs[intersect]
                if intersect_trj.intersect_count < self.min_group_trj_nums or intersect_trj.size < self.min_lifetime:
                    continue
                # if trj.token_seq[0] >= intersect_trj.token_seq[-self.min_lifetime] or intersect_trj.token_seq[0] >= trj.token_seq[-self.min_lifetime]:
                #     continue
                # cvals = list(set(trj.token_seq).intersection(set(intersect_trj.token_seq)))
                cvals = list((Counter(trj.token_seq) & Counter(intersect_trj.token_seq)).elements())
                if len(cvals) >= self.min_lifetime:
                    pairs.append([trj.id, intersect_trj.id])
                    count += 1
                if count + trj.intersect_count - ii < self.min_group_trj_nums:
                    break
            if count >= self.min_group_trj_nums:
                trj.candiate_match = pairs
                all_pairs.extend(pairs)
        print("There are totally {} valid candidate groups".format(len(all_pairs)))
        return all_pairs

    def get_groups(self, trjs, ep):
        # step 1
        pairs = self.get_pairs(trjs)
        # step 2
        communities, hubs, outliers = get_communities(pairs, self.min_group_trj_nums, ep)
        labels = [-1] * len(trjs)
        for ii in range(len(communities)):
            for trj_id in communities[ii]:
                labels[trj_id] = ii
        print(
            '归组人数:{0}\t组数：{1}'.format(sum(np.array(labels) != -1), len(communities)))
        return pairs, communities, labels






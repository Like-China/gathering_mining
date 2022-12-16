import time
import numpy as np
import settings as constants
from tqdm import tqdm
from index.scan import  get_communities


class ACMC:

    def __init__(self):
        self.min_group_trj_nums = constants.min_group_trj_nums
        self.min_lifetime = constants.min_lifetime

    ''' 
        计算得到全部的group
        利用LCS+SCAN计算群组网络，返回哪些人是一个群组
        1. 记录哪些轨迹两两的LCS大于min_lifetime
        2. 连线这些轨迹并进行SCAN聚类
    '''

    def get_groups(self, all_trjs):
        # step 1
        connectable_trj_pair = []
        for i in tqdm(range(len(all_trjs))):
            for j in range(i + 1, len(all_trjs)):
                trj1 = all_trjs[i]
                trj2 = all_trjs[j]
                if len(trj1)<self.min_lifetime or len(trj2)<self.min_lifetime:
                    continue
                if trj1[0] >= trj2[-self.min_lifetime] or trj2[0] >= trj1[-self.min_lifetime]:
                    continue
                cvals = list(set(trj1).intersection(set(trj2)))
                if len(cvals) >= self.min_lifetime:
                    connectable_trj_pair.append([i, j])
        # step 2
        communities, hubs, outliers = get_communities(connectable_trj_pair, self.min_group_trj_nums, constants.ep)
        trj2community = [-1] * len(all_trjs)
        for ii in range(len(communities)):
            for trj_id in communities[ii]:
                trj2community[trj_id] = ii
        print(
            '归组人数:{0}\t未归组人数：{1}\t组数：{2}'.format(sum(np.array(trj2community) != -1), sum(np.array(trj2community) == -1),
                                                 len(communities)))
        return trj2community


if __name__ == "__main__":
    a = ACMC()
    # Tests whether the search group is correct
    sequences = []
    for ii in range(5000):
        size = np.random.randint(0, 20) + 3
        sequence = np.random.randint(0, 20, size).tolist()
        sequence = sorted(sequence)
        sequences.append(sequence)
    trj2community = a.get_groups(sequences)




import numpy as np
from tqdm import tqdm


class ECMC:

    def __init__(self, min_group_trj_nums, min_lifetime, dist_error, time_error):
        self.min_group_trj_nums = min_group_trj_nums
        self.min_lifetime = min_lifetime
        self.dist_error = dist_error
        self.time_error = time_error

    def get_pairs(self, trjs):
        """
        得到每条轨迹可以和哪些轨迹形成有效的轨迹对
        :param seqs: a batch of trajectories token sequences
        :return: all of the co-movement pairs [[i,j, common_points]]
        """
        # t1 = time.time()
        all_pairs = []
        for trj in tqdm(trjs, desc='get pairs'):
            pairs = [trj]
            count = 0
            if trj.intersect_count < self.min_group_trj_nums:
                continue
            for ii, intersect in enumerate(trj.intersect_trjs):
                if intersect == trj.id:
                    continue
                intersect_trj = trjs[intersect]
                if intersect_trj.intersect_count < self.min_group_trj_nums:
                    continue
                # if intersect_trj in trj.candiate_match:
                #     pairs.append(intersect_trj)
                #     count += 1
                #     continue
                lcs = trj.LCS_to(intersect_trj, self.min_lifetime, self.dist_error, self.time_error)
                if lcs >= self.min_lifetime:
                    pairs.append(intersect_trj)
                    count += 1
                if count + trj.intersect_count - ii < self.min_group_trj_nums:
                    break
            if count >= self.min_group_trj_nums:
                trj.candiate_match = pairs
                all_pairs.append(pairs)
        print("There are totally {} valid candidate groups".format(len(all_pairs)))
        return all_pairs

    def get_groups(self, trjs):
        """
        get this batch trajectories' companion pairs, companion groups, companion trj2group
        :param seqs:
        :return:
        """
        all_pairs = self.get_pairs(trjs)
        all_groups = []
        # 记录已经归组的轨迹
        grouped_id = [-1]*len(trjs)
        for pairs in all_pairs:
            # pairs [trj, trj1,trj2,...]
            valid_range = max(len(pairs) - self.min_group_trj_nums, 1)
            # 记录最大长度，用于选取最合适的聚类组合
            max_len = 0
            best_group = []
            for ii in range(1, valid_range):
                if grouped_id[pairs[0].id] == 1:
                    break
                if grouped_id[pairs[ii].id] == 1:
                    continue
                group = [pairs[0], pairs[ii]]
                for jj in range(ii + 1, len(pairs)):
                    check_trj = pairs[jj]
                    if grouped_id[check_trj.id] == 1:
                        continue
                    flag = True
                    for trj in group:
                        if check_trj not in trj.candiate_match:
                            flag = False
                            break
                    if flag: group.append(check_trj)
                size = len(group)
                if size >= max_len:
                    max_len = size
                    best_group = group
            # 只记录成员数最多的聚类组合, 并标记该组合内成员已经标记
            if len(best_group) >= self.min_group_trj_nums:
                all_groups.append(best_group)
                for trj in best_group:
                    grouped_id[trj.id] = 1

        trj_map_group = [-1] * len(trjs)
        for ii, group in enumerate(all_groups):
            for trj in group:
                trj.cluster_id = ii
                trj_map_group[trj.id] = ii

        # print('Number of trajectory not in the group：{} group number：{}'.format(sum(np.array(trj_map_group) == -1),len(all_groups)))
        # for pairs in all_pairs:
        #     print([trj.id for trj in pairs])
        # print("\n")
        # for group in all_groups:
        #     print([trj.id for trj in group])
        return all_pairs, all_groups, trj_map_group



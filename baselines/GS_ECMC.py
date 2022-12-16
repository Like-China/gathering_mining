import time
from funcy import merge
import numpy as np
import settings as constants
import settings as settings
from tqdm import tqdm


class ECMC:

    def __init__(self):
        self.min_group_trj_nums = constants.min_group_trj_nums
        self.min_lifetime = constants.min_lifetime

    def is_match(self, point_a, point_b):

        if point_a[2] != point_b[2]:
            return False
        if abs(point_a[0] - point_b[0]) > settings.scale/10:
            return False
        if abs(point_a[1] - point_b[1]) > settings.scale/10:
            return False
        return True
    
    def common_points(self, trajectory_a, trajectory_b):
        len_m = len(trajectory_a)
        len_n = len(trajectory_b)
        min_length = min(len_m, len_n)
        if min_length < self.min_lifetime:
            return []
        i, j = 0, 0
        common_point = []
        while i < len_m and j < len_n:
            if trajectory_a[i][2] == trajectory_b[j][2]:
                if self.is_match(trajectory_a[i], trajectory_b[j]):
                    common_point.append(trajectory_a[i])
                i += 1
                j += 1
            elif trajectory_a[i][2] < trajectory_b[j][2]:
                i += 1
            else:
                j += 1
        return common_point

    def get_pairs(self, seqs):
        """
        Enter a set of token sequences to obtain all co-movement pairs
        Note: ：A trajectory can pair with multiple trajectories
        :param seqs: a batch of trajectories token sequences
        :return: all of the co-movement pairs [[i,j, common_points]]
        """
        # t1 = time.time()
        all_pairs = []
        trj_num = len(seqs)
        for i in tqdm(range(trj_num), desc='get pairs'):
            pairs = []
            for j in range(i + 1, trj_num):
                common_point = self.common_points(seqs[i], seqs[j])
                if len(common_point) >= self.min_lifetime:
                    pairs.append([i, j, common_point])
            if len(pairs) >= self.min_group_trj_nums:
                all_pairs.append(pairs)
        return all_pairs

    def get_groups(self, seqs):
        """
        get this batch trajectories' companion pairs, companion groups, companion trj2group
        :param seqs:
        :return:
        """
        all_pairs = self.get_pairs(seqs)
        print(len(all_pairs))
        all_groups = []
        grouped_set = set()
        for pairs in all_pairs:
            for ii in range(max(len(pairs) - self.min_group_trj_nums, 1)):
                if pairs[ii][0] in grouped_set:
                    continue  
                group = [pairs[ii][0], pairs[ii][1]]
                common_point = pairs[ii][2]
                for jj in range(ii + 1, len(pairs)):
                    if pairs[jj][1] in grouped_set:
                        continue  
                    trj2 = pairs[jj][2]
                    common_point = self.common_points(common_point, trj2)
                    if len(common_point) >= self.min_lifetime:
                        group.append(pairs[jj][1])
                if len(group) >= self.min_group_trj_nums:
                    all_groups.append(group)
                    grouped_set.update(set(group))
                    break
        trj_map_group = [-1] * len(seqs)
        for ii in range(len(all_groups)):
            for trj_id in all_groups[ii]:
                trj_map_group[trj_id] = ii

        n1 = sum(np.array(trj_map_group) != -1)
        n2 = sum(np.array(trj_map_group) == -1)
        print('The number of belong a group:{0} Number of trajectory not in the group：{1} group number：{2}'.format(n1, n2,len(all_groups)))
        if len(all_pairs) > 0:
            all_pairs = merge(*all_pairs)
        return all_pairs, all_groups, trj_map_group


if __name__ == "__main__":

    e = ECMC()
    trajectories = []

    for ii in range(10000):
        trajectory = []
        t = sorted(np.random.randint(0, 10, 100).tolist())
        for jj in range(10):
            lon = np.random.random()*settings.scale
            lat = np.random.random()*settings.scale
            trajectory.append([lon, lat, 10])
        trajectories.append(trajectory)

    # for ii in tqdm(range(len(trajectories)-1)):
    #     trajectory_a = trajectories[ii]
    #     trajectory_b = trajectories[ii+1]
    #     common_point = e.common_points(trajectory_a, trajectory_b)
    # print(common_point)
    import time
    t1 = time.time()
    all_pairs, all_groups, trj_map_group = e.get_groups(trajectories)
    print(time.time()-t1)
    # print(all_pairs)
    # print(all_groups)
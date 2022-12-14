import time
import numpy as np
import settings as constants


class ACMC:

    def __init__(self):
        self.min_group_trj_nums = constants.min_group_trj_nums
        self.min_lifetime = constants.min_lifetime

    def common_values(self, m, n):
        len_m = len(m)
        len_n = len(n)
        min_length = min(len_m, len_n)
        if min_length < self.min_lifetime: 
            return []
        if m[0] >= n[-self.min_lifetime] or n[0] >= m[-self.min_lifetime]:
            return []
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

    def get_pairs(self, seqs):
        t1 = time.time()
        all_pairs = []
        trj_num = len(seqs)
        for i in range(trj_num):
            pairs = []
            for j in range(i + 1, trj_num):
                common_value = self.common_values(seqs[i], seqs[j])
                if len(common_value) >= self.min_lifetime:
                    pairs.append([i, j, common_value])
            if len(pairs) >= self.min_group_trj_nums:
                all_pairs.append(pairs)
        print('the time-consuming of getting all pairs:' + str(time.time() - t1))
        return all_pairs

    def get_groups(self, seqs):
        all_pairs = []
        all_groups = []
        grouped_set = set()
        for i in range(len(seqs)):
            if i in grouped_set: 
                continue  
            pairs = []
            for j in range(i + 1, len(seqs)):
                if j in grouped_set: 
                    continue  
                common_value = self.common_values(seqs[i], seqs[j])
                if len(common_value) >= self.min_lifetime:
                    pairs.append([i, j, common_value])

            if len(pairs) >= self.min_group_trj_nums:
                all_pairs.append(pairs)
                for ii in range(max(len(pairs) - self.min_group_trj_nums, 1)):
                    if pairs[ii][0] in grouped_set:
                        continue  
                    group = [pairs[ii][0], pairs[ii][1]]
                    common_value = pairs[ii][2]
                    for jj in range(ii + 1, len(pairs)):
                        if pairs[jj][1] in grouped_set:
                            continue  
                        trj2 = pairs[jj][2]
                        common_value = self.common_values(common_value, trj2)

                        if len(common_value) >= self.min_lifetime:
                            group.append(pairs[jj][1])
                    if len(group) >= self.min_group_trj_nums:
                        # print(group)
                        # for pair in pairs:
                        #     print(pair)
                        # cv = seqs[group[0]]
                        # for id in group:
                        #     cv = self.common_values(cv, seqs[id])
                        # print(cv)
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
            all_pairs = sum(all_pairs, [])
        return all_pairs, all_groups, trj_map_group

    def get_groups1(self, seqs):
        """
        get this batch trajectories' companion pairs, companion groups, companion trj2group
        :param seqs:
        :return:
        """
        all_pairs = self.get_pairs(seqs)
        all_groups = []
        grouped_set = set()
        for pairs in all_pairs:
            for ii in range(max(len(pairs) - self.min_group_trj_nums, 1)):
                if pairs[ii][0] in grouped_set:
                    continue
                group = [pairs[ii][0], pairs[ii][1]]
                common_value = pairs[ii][2]
                for jj in range(ii + 1, len(pairs)):
                    if pairs[jj][1] in grouped_set:
                        continue 
                    trj2 = pairs[jj][2]
                    common_value = self.common_values(common_value, trj2)
                    if len(common_value) >= self.min_lifetime:
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
            all_pairs = sum(all_pairs, [])
        return all_pairs, all_groups, trj_map_group


if __name__ == "__main__":
    a = ACMC()
    s1 = [153219, 157939, 160299, 162659, 167379, 169739, 174459, 176819]
    s2 = [131979, 131979, 131979, 134339, 136699, 136699, 139059, 141419, 141419, 143779, 146139, 146139, 148499, 150859, 153219, 153219, 155579, 157939, 157939, 160299, 162659, 162659, 165019, 167379, 167379, 169739, 172099, 174459, 174459, 176819, 179179, 179179, 179179, 181539, 183899, 183723, 185903, 190325, 192087]
    cv = a.common_values(s1, s2)
    print(cv)

    # Tests whether the search group is correct
    sequences = []
    for ii in range(50):
        size = np.random.randint(0, 10) + 3
        sequence = np.random.randint(0, 5, size).tolist()
        sequence = sorted(sequence)
        sequences.append(sequence)
    all_pairs, all_groups, trj_map_group = a.get_groups(sequences)



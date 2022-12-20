from loader.data_loader import Loader
from indexing.myRtree import my_rtree
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from baselines.GS_ECMC_multi import ECMC
from baselines.LCS_SCAN import SCAN
from baselines.GS_ACMC import ACMC
import settings as ss


def recall(real_label, pred_label):
    hit_count = 0
    unhit_count = 0
    for ii in  range(len(real_label)):
        for jj in range(ii+1, len(real_label)):
            if real_label[ii] == -1 or real_label[jj] == -1:
                continue
            if real_label[ii] == real_label[jj]:
                if pred_label[ii] == pred_label[jj] and pred_label[ii] != -1:
                    hit_count += 1
                else:
                    unhit_count += 1
    if (hit_count+unhit_count) == 0:
        return 0
    else:
        # print(hit_count, unhit_count)
        return hit_count/(hit_count+unhit_count)


def hit_ratio(real_label, pred_label):
    TP, TN, FP, FN = 0, 0, 0, 0
    for ii in  range(len(real_label)):
        for jj in range(ii+1, len(real_label)):
            # if real_label[jj] == -1:
            #     continue
            if real_label[ii] == real_label[jj]:
                if pred_label[ii] == pred_label[jj]:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred_label[ii] == pred_label[jj]:
                    FP += 1
                else:
                    TN += 1
    if (TP+FN) == 0:
        return 0
    else:
        print(TP, TP+FN, FP)
        return TP/(TP+FN)


def load_index(scale, time_size, num):
    # 加载数据
    trajectory_set = Loader(scale, time_size).load(num)
    # 建立索引
    idx = my_rtree()
    for trj in trajectory_set:
        idx.insert(trj.id, trj.mbr)
    # 索引查询, 对每条轨迹记录其有时空交集的其他轨迹id序列
    for trj in trajectory_set:
        trj.set_intersect_trjs(list(idx.intersection(trj.mbr)))
    return trajectory_set, idx

import datetime
def find_best(nums, time_sizes, scales, eps, dist_errors, time_errors):
    res = []
    for num in nums:
        for dist_error in dist_errors:
            for time_error in time_errors:
                for min_lifetime in [3,4,5,6,7]:
                    print(datetime.datetime.now())
                    trajectory_set, idx = load_index(scales[0], time_sizes[0], num)
                    # 基于轨迹对象，开始执行查询, 精确查询 Ground-truth
                    ECMC_all_pairs, ECMC_all_groups,  ecmc_labels = ECMC(min_lifetime, min_lifetime, dist_error, time_error).get_groups(
                        trajectory_set)
                    for time_size in time_sizes:
                        for scale in scales:
                            print(
                                "\n***************num={}, time_size={}, scale={}, dist_error={}, time_error={}, life={}****************".format(
                                    num, time_size, scale, dist_error, time_error, min_lifetime))
                            # LCS+SCAN 快速组合
                            trajectory_set, idx = load_index(scale, time_size, num)
                            SCAN_all_pairs, SCAN_all_groups, scan_labels = SCAN(min_lifetime, min_lifetime).get_groups(trajectory_set, 0.5)
                            r = recall(ecmc_labels, scan_labels)
                            print("Recall: {}".format(r))
                            res.append([num, time_size, scale, min_lifetime, dist_error, time_error,
                                        len(ECMC_all_pairs),  len(SCAN_all_pairs),
                                        len(ECMC_all_groups),  len(SCAN_all_groups), r])
                            np.save("res", res)


class Evaluator:
    def __init__(self, n):
        self.n = 10

    def vary_nums(self):
        res = []
        times = []
        for num in ss.nums:
            print("num: {}".format(num))
            # 随机加载数据
            trajectory_set, idx = load_index(ss.scale, ss.time_size, num)
            t1 = time.time()
            e_pairs, e_groups, e_labels = ECMC(ss.min_lifetime, ss.min_group_trj_num, ss.dist_error,
                                                                ss.time_error).get_groups(
                trajectory_set)
            t2 = time.time()
            trajectory_set, idx = load_index(ss.scale, ss.time_size, num)
            t3 = time.time()
            s_pairs, s_groups, s_labels = SCAN(ss.min_lifetime, ss.min_group_trj_num).get_groups(
                trajectory_set, ss.ep)
            t4 = time.time()
            times.append([t2 - t1, t4 - t3])
            r = recall(e_labels, s_labels)
            print("Recall: {}".format(r))
            res.append(r)
        print(res, times)


    def vary_min_life_times(self):
        res = []
        times = []
        for min_lifetime in ss.min_life_times:
            print("min_lifetime: {}".format(min_lifetime))
            # 随机加载数据
            trajectory_set, idx = load_index(ss.scale, ss.time_size, ss.num)
            t1 = time.time()
            e_pairs, e_groups, e_labels = ECMC(min_lifetime, ss.min_group_trj_num, ss.dist_error,
                                               ss.time_error).get_groups(
                trajectory_set)
            t2 = time.time()
            trajectory_set, idx = load_index(ss.scale, ss.time_size, ss.num)
            t3 = time.time()
            s_pairs, s_groups, s_labels = SCAN(min_lifetime, ss.min_group_trj_num).get_groups(
            trajectory_set, ss.ep)
            t4 = time.time()
            times.append([t2-t1,t4-t3])
            r = recall(e_labels, s_labels)
            print("Recall: {}".format(r))
            res.append(r)
        print(res, times)

    def vary_min_group_trj_nums(self):
        res = []
        times = []
        for min_group_trj_num in ss.min_group_trj_nums:
            print("min_group_trj_num: {}".format(min_group_trj_num))
            # 随机加载数据
            trajectory_set, idx = load_index(ss.scale, ss.time_size, ss.num)
            t1 = time.time()
            e_pairs, e_groups, e_labels = ECMC(ss.min_lifetime, min_group_trj_num, ss.dist_error,
                                               ss.time_error).get_groups(
                trajectory_set)
            t2 = time.time()
            trajectory_set, idx = load_index(ss.scale, ss.time_size, ss.num)
            t3 = time.time()
            s_pairs, s_groups, s_labels = SCAN(ss.min_lifetime, min_group_trj_num).get_groups(
            trajectory_set, ss.ep)
            t4 = time.time()
            times.append([t2 - t1, t4 - t3])
            r = recall(e_labels, s_labels)
            print("Recall: {}".format(r))
            res.append(r)
        print(res, times)

    def vary_eps(self):
        res = []
        times = []
        for ep in ss.eps:
            print("ep: {}".format(ep))
            # 随机加载数据
            trajectory_set, idx = load_index(ss.scale, ss.time_size, ss.num)
            t1 = time.time()
            e_pairs, e_groups, e_labels = ECMC(ss.min_lifetime, ss.min_group_trj_num, ss.dist_error,
                                               ss.time_error).get_groups(
                trajectory_set)
            t2 = time.time()
            trajectory_set, idx = load_index(ss.scale, ss.time_size, ss.num)
            t3 = time.time()
            s_pairs, s_groups, s_labels = SCAN(ss.min_lifetime, ss.min_group_trj_num).get_groups(
            trajectory_set, ep)
            t4 = time.time()
            times.append([t2 - t1, t4 - t3])
            r = recall(e_labels, s_labels)
            print("Recall: {}".format(r))
            res.append(r)
        print(res, times)




if __name__ == "__main__":
    # res = np.load("res.npy", allow_pickle=True)
    # import pandas as pd
    # res = pd.DataFrame(res, columns=['num','time','scale','min_lifetime','derror','terror','epair','spair','eg','sg','r'])
    # res.to_csv("res20000.csv")
    # nums, time_sizes, scales = [10000,20000], [40], [0.001,0.002]
    # dist_errors = np.linspace(100,300,3).tolist()
    # time_errors = np.linspace(100,300,3).tolist()
    # eps = [0.5]
    # find_best(nums, time_sizes, scales, eps, dist_errors, time_errors)

    # evaluate
    E = Evaluator(ss.n)
    # [0.7509117883414724, 0.7239042310615658, 0.7000438581694275, 0.6921727050049354, 0.6970669110907425]
    # [[611.9157228469849, 31.511074781417847], [608.7182819843292, 31.491008281707764], [605.2445735931396, 31.557518243789673], [602.0628037452698, 31.73237442970276], [602.0014488697052, 31.048818588256836]]
    # E.vary_min_life_times()
    # [0.8804949381327334, 0.7894880248494591, 0.7001450693296447, 0.634835688602726, 0.589820664357041]
    # [[635.3509919643402, 35.28736352920532], [616.8663096427917, 32.29002928733826], [596.3034536838531, 31.094136953353882], [600.8406708240509, 36.155229568481445], [827.7463233470917, 32.041985750198364]]
    # E.vary_min_group_trj_nums()
    # [0.92051550217604, 0.8805708309436254, 0.8360041833946223, 0.7719375189770925, 0.6997064876353699]
    # [[711.6715259552002, 38.220566511154175], [839.027574300766, 38.68840503692627], [834.9736180305481, 37.431989908218384], [801.4750401973724, 38.27264881134033], [817.7547326087952, 39.402512073516846]]
    # E.vary_eps()
    # [0.699976384062616, 0.768532950932103, 0.8229892519278994, 0.8545432343934267, 0.8750129455424183] Porto
    # [[865.1456763744354, 38.56213092803955], [3088.8664219379425, 162.89063692092896], [5761.950331687927, 397.9759519100189], [13502.952735185623, 748.6925759315491], [20606.409828662872, 1188.985583305359]]
    E.vary_nums()
import settings
from indexing.dist import get_distance_hav
import numpy as np


class Trajectory:

    def __init__(self, id):
        self.id = id
        self.lon_lat_seq = []
        self.time_seq = []
        self.mbr = [-1000,1000,-1000,1000,-1000,1000]
        self.time_range = None
        self.token_seq = []
        self.intersect_trjs = []
        self.intersect_count = 0
        self.size = 0
        # 以该轨迹为扩张原点，可以和该轨迹形成有效pair的轨迹
        self.candiate_match = []
        self.cluster_id = -1


    def set_intersect_trjs(self, intersect_trjs):
        self.intersect_trjs = intersect_trjs
        self.intersect_count = len(intersect_trjs)

    def set_lon_lat(self, lon_lat_seq):
        self.lon_lat_seq = lon_lat_seq
        min_lon, min_lat = np.min(lon_lat_seq, axis=0)
        max_lon, max_lat = np.max(lon_lat_seq, axis=0)
        self.mbr = [min_lon, max_lon, min_lat, max_lat, -1000,1000]
        self.size = len(lon_lat_seq)
        # get first column
        # arr[:, 0]

    def set_token_seq(self, token_seq):
        self.token_seq = token_seq

    def set_time_seq(self, time_seq):
        self.time_seq = time_seq
        self.time_range = [time_seq[0], time_seq[-1]]
        self.mbr[4], self.mbr[5] = time_seq[0], time_seq[-1]

    def __len__(self):
        return self.size

    def __str__(self):
        return str(self.lon_lat_seq.tolist()) + "\n"+ str(self.time_seq.tolist()) + "\n"+ str(self.token_seq) + "\n"+ str(self.mbr)

    def LCS_to(self, trj, min_lifetime, dist_error, time_error):
        m = self.size
        n = trj.size
        # 定义一个列表来保存最长公共子序列的长度，并初始化
        record = [[0 for i in range(n + 1)] for j in range(m + 1)]
        if trj.time_seq[0]>self.time_seq[-min_lifetime] or self.time_seq[0]>trj.time_seq[-min_lifetime]:
            return 0
        for i in range(m):
            for j in range(n):
                lon1, lat1, time1 = self.lon_lat_seq[i][0], self.lon_lat_seq[i][1], self.time_seq[i]
                lon2, lat2, time2 = trj.lon_lat_seq[j][0], trj.lon_lat_seq[j][1], trj.time_seq[j]
                if get_distance_hav(lon1, lat1, lon2, lat2) <= dist_error and abs(time1-time2)<=time_error:
                    record[i + 1][j + 1] = record[i][j] + 1
                elif record[i + 1][j] > record[i][j + 1]:
                    record[i + 1][j + 1] = record[i + 1][j]
                else:
                    record[i + 1][j + 1] = record[i][j + 1]
        return record[-1][-1]


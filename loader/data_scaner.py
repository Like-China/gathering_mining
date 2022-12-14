"""
At test time, test set tracks are read sequentially, batch by batch
"""
from loader.data_utils import *


class DataOrderScaner():

    def __init__(self, srcfile):
        self.srcfile = srcfile
        self.srcdata = []
        self.start = 0

    def load(self, max_num_line=0):
        num_line = 0
        with open(self.srcfile, 'r') as srcstream:
            for s in srcstream:
                s = [int(x) for x in s.split()]
                self.srcdata.append(np.array(s, dtype=np.int32).tolist())
                num_line += 1
                if max_num_line > 0 and num_line >= max_num_line:
                    break
        self.size = len(self.srcdata)
        self.start = 0

    ''' Load data based on time periods'''
    def load_by_period(self):
        self.peak_trjs = []
        self.work_trjs = []
        self.casual_trjs = []
        for trj in self.srcdata:
            if self.period(trj) == 1:
                self.peak_trjs.append(trj)
            elif self.period(trj) == 2:
                self.work_trjs.append(trj)
            else:
                self.casual_trjs.append(trj)
        print("peak_num:{}, work_num:{},casual_num:{}".format(
            len(self.peak_trjs), len(self.work_trjs), len(self.casual_trjs)))
        return pad_arrays_keep_invp(self.peak_trjs), pad_arrays_keep_invp(self.work_trjs), \
               pad_arrays_keep_invp(self.casual_trjs)

    ''' Calculates the time period to which the trajectory belongs '''

    def period(self, trj):
        cityname = constants.cityname
        scale = constants.scale
        time_size = constants.time_size
        if cityname == "beijing":
            lons_range, lats_range = constants.lons_range_bj, constants.lats_range_bj
        else:
            lons_range, lats_range = constants.lons_range_pt, constants.lats_range_pt

        '''Gets the partition of the space under the current scale setting'''
        maxx, maxy = (lons_range[1] - lons_range[0]) // scale, (lats_range[1] - lats_range[0]) // scale
        space_size = maxx * maxy
        time1 = trj[0] // space_size
        time1 = (24 * time1) // time_size
        time2 = trj[-1] // space_size
        time2 = (24 * time2) // time_size
        if time1 >= 7 and time1 <= 10 and time2 >= 7 and time2 <= 10:
            return 1
        elif time1 >= 17 and time1 <= 20 and time2 >= 17 and time2 <= 20:
            return 1
        elif time1 >= 10 and time1 <= 17 and time2 >= 10 and time2 <= 17:
            return 2
        else:
            return 3

    '''Get a fixed number of trajectories'''

    def getbatch(self, batch):
        """
        Output:
        src (seq_len, batch)
        lengths (1, batch)
        invp (batch,): inverse permutation, src.t()[invp] gets original order
        """
        if self.start >= self.size:
            return None, None, None
        src = self.srcdata[self.start:self.start + batch]
        # update `start` for next batch
        self.start += self.batch
        self.start = 0
        return pad_arrays_keep_invp(src)





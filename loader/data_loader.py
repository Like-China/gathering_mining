# -*- coding: utf-8 -*-
import os
import numpy as np
import warnings, argparse, h5py, time
import settings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.trajectory import Trajectory


def set_region_args(scale, time_size):

    parser = argparse.ArgumentParser(description="Region.py")
    parser.add_argument("-city", default=settings.city, help="city name")
    if settings.city == "beijing":
        lons_range, lats_range = settings.lons_range_bj, settings.lats_range_bj
    else:
        lons_range, lats_range = settings.lons_range_pt, settings.lats_range_pt
    minx, miny = 0, 0
    maxx, maxy = (lons_range[1]-lons_range[0])//scale, (lats_range[1]-lats_range[0])//scale
    # Spatial partition parameter
    parser.add_argument("-lons", default=lons_range, help="Range of longitude")
    parser.add_argument("-lats", default=lats_range, help="Range of latitude")
    parser.add_argument("-scale", default=scale, help="Space cell size")
    parser.add_argument("-minx", type=int, default=minx, help="the number of minimum x-coordinate")
    parser.add_argument("-maxx", type=int, default=maxx, help="the number of maximum x-coordinate")
    parser.add_argument("-miny", type=int, default=miny, help="the number of minimum y-coordinate")
    parser.add_argument("-maxy", type=int, default=maxy, help="the number of maximum y-coordinate")
    parser.add_argument("-numx", type=int, default=maxx, help="Number of horizontal blocks in space")
    parser.add_argument("-numy", type=int, default=maxy, help="Number of vertical blocks in space")
    parser.add_argument("-space_cell_size", type=int, default=maxx*maxy, help="space_cell_size")
    # Time division parameter
    parser.add_argument("-time_size", type=int, default=time_size, help="Number of time periods in a day")
    parser.add_argument("-time_span", type=int, default=86400 // time_size, help="Length of each time period")
    # Spatio-Temporal grid parameter
    parser.add_argument("-map_cell_size", type=int, default=maxx*maxy*time_size, help="Number of space-time cells (x,y,t) three dimensions")
    args = parser.parse_args()
    return args


class Loader:

    def __init__(self, scale, time_size):
        self.args = set_region_args(scale, time_size)
        self.h5path = os.path.join("/data/Like/", self.args.city + ".h5")
        # self.h5path = os.path.join("E:\\data\porto.h5")
        # print("parameter setting： \n", self.args)

    # load trajectory instance, return a set of trajectories
    def load(self, read_trj_num):
        f = h5py.File(self.h5path, 'r')
        trj_nums = min(f.attrs['num'], read_trj_num)
        trajectory_set = [Trajectory(ii) for ii in range(trj_nums)]
        # for i in tqdm(range(trj_nums), desc='read lon, lat'):
        for i in range(trj_nums):
            trip = np.array(f.get('trips/%d' % (i+1)))[:settings.max_len] # numpy n*2, [[lon,lat],[lon,lat]]
            trajectory_set[i].set_lon_lat(trip)
        # for i in tqdm(range(trj_nums), desc='read timestamp'):
        for i in range(trj_nums):
            ts = np.array(f.get('timestamps/%d' % (i+1)))[:settings.max_len] # numpy n*1
            trajectory_set[i].set_time_seq(ts)
        for i in range(trj_nums):
        # for i in tqdm(range(trj_nums), desc='get token seqs'):
            trj = trajectory_set[i]
            trj.token_seq = self.trip2mapIDs(trj.lon_lat_seq, trj.time_seq)[:settings.max_len]
            # trj.token_seq = self.trip2spaceIDs(trj.lon_lat_seq)
        return trajectory_set

    def observe(self, raw_trj, map_ids):
        plt.figure()
        for point in raw_trj:
            plt.scatter(point[0], point[1])
        plt.show()
        plt.figure()
        for id in map_ids:
            spid = self.mapId2spaceId(id)
            x, y = self.spaceId2offset(spid)
            plt.scatter(x, y)
        plt.show()

    '''
        ****************************************************************************************************************************************************
        A bunch of transformation functions
        ****************************************************************************************************************************************************
        '''
    '''Latitude and longitude are converted into meters and mapped onto the plane plan (116.3, 40.0)->(4,8)'''

    def lonlat2xyoffset(self, lon, lat):
        x_offset = round((lon - self.args.lons[0]) / self.args.scale)
        y_offset = round((lat - self.args.lats[0]) / self.args.scale)
        return int(x_offset), int(y_offset)

    ''' Meters convert to latitude and longitude  (4,8)-> (116.3, 40.0)'''

    def xyoffset2lonlat(self, x_offset, y_offset):
        lon = self.args.lons[0] + x_offset * self.args.scale
        lat = self.args.lats[0] + y_offset * self.args.scale
        return lon, lat

    ''' (x_offset,y_offset) -> space_cell_id  (4,8)->116'''

    def offset2spaceId(self, x_offset, y_offset):
        return int(y_offset * self.args.numx + x_offset)

    ''' space_cell_id -->(x,y) 116->(4.8)'''

    def spaceId2offset(self, space_cell_id):
        y_offset = space_cell_id // self.args.numx
        x_offset = space_cell_id % self.args.numx
        return int(x_offset), int(y_offset)

    ''' gps--> space_cell_id  116.3,40->116'''

    def gps2spaceId(self, lon, lat):
        x_offset, y_offset = self.lonlat2xyoffset(lon, lat)
        space_cell_id = self.offset2spaceId(x_offset, y_offset)
        return int(space_cell_id)

    '''space_cell_id -->gps 116->116.3,40'''

    def spaceId2gps(self, space_cell_id):
        x_offset, y_offset = self.spaceId2offset(space_cell_id)
        lon, lat = self.xyoffset2lonlat(x_offset, y_offset)
        return lon, lat

    ''' space_cell_id+t --> map_id  116,10->1796'''

    def spaceId2mapId(self, space_id, t):
        return int(space_id + t * self.args.space_cell_size)

    ''' map_id -->space_cell_id  1796-> 116'''

    def mapId2spaceId(self, map_id):
        return int(map_id % self.args.space_cell_size)

    def trip2spaceIDs(self, trip):
        space_ids = []
        for (lon, lat) in trip:
            space_id = self.gps2spaceId(lon, lat)
            space_ids.append(space_id)
        return space_ids

    def trip2mapIDs(self, trip, ts):
        map_ids = []
        for (lon, lat), t in zip(trip, ts):
            space_id = self.gps2spaceId(lon, lat)
            t = int(t) // self.args.time_span
            map_id = int(self.spaceId2mapId(space_id, t))
            map_ids.append(map_id)
        return map_ids


if __name__ == "__main__":
    r = Loader()
    trajectory_set = r.load(10000)
    trj = trajectory_set[0]
    print(trj)
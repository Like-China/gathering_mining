# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 17:46:09 2021

@author: likem

Each trajectory point is converted into token value by dividing space and time
Read the trajectory of the h5 files, Write in 
./data/city_name/cityname_regionScale_timeScale/train.src
./data/city_name/cityname_regionScale_timeScale/train1.trg
./data/city_name/cityname_regionScale_timeScale/val.src
./data/city_name/cityname_regionScale_timeScale/val.trg
"""
import os
import numpy as np
import warnings, argparse, h5py, time
import settings
warnings.filterwarnings("ignore")


def set_region_args():

    parser = argparse.ArgumentParser(description="Region.py")
    parser.add_argument("-city", default=settings.city, help="city name")
    if settings.city == "beijing":
        lons_range, lats_range = settings.lons_range_bj, settings.lats_range_bj
    else:
        lons_range, lats_range = settings.lons_range_pt, settings.lats_range_pt
    minx, miny = 0, 0
    maxx, maxy = (lons_range[1]-lons_range[0])//settings.scale, (lats_range[1]-lats_range[0])//settings.scale
    # Spatial partition parameter
    parser.add_argument("-lons", default=lons_range, help="Range of longitude")
    parser.add_argument("-lats", default=lats_range, help="Range of latitude")
    parser.add_argument("-scale", default=settings.scale, help="Space cell size")
    parser.add_argument("-minx", type=int, default=minx, help="the number of minimum x-coordinate")
    parser.add_argument("-maxx", type=int, default=maxx, help="the number of maximum x-coordinate")
    parser.add_argument("-miny", type=int, default=miny, help="the number of minimum y-coordinate")
    parser.add_argument("-maxy", type=int, default=maxy, help="the number of maximum y-coordinate")
    parser.add_argument("-numx", type=int, default=maxx, help="Number of horizontal blocks in space")
    parser.add_argument("-numy", type=int, default=maxy, help="Number of vertical blocks in space")
    parser.add_argument("-space_cell_size", type=int, default=maxx*maxy, help="space_cell_size")
    # Time division parameter
    parser.add_argument("-time_size", type=int, default=settings.time_size, help="Number of time periods in a day")
    parser.add_argument("-time_span", type=int, default=86400 // settings.time_size, help="Length of each time period")
    # Spatio-Temporal grid parameter
    parser.add_argument("-start", type=int, default=settings.START, help="Vocal word code is numbered from 4, 0,1,2,3 have special functions")
    parser.add_argument("-map_cell_size", type=int, default=maxx*maxy*settings.time_size, help="Number of space-time cells (x,y,t) three dimensions")
    args = parser.parse_args()
    return args


class Region:
    '''
        Divide space grid and time grid
        For each original (x,y,t) trajectory point, it is transformed into the corresponding space-time grid code
        input：
        args  Set training parameters
        output：
        ./data/city/cityname_regionScale_timeScale/train.src
        ./data/city/cityname_regionScale_timeScale/val.src
    '''

    def __init__(self):
        self.args = set_region_args()
        
        self.save_path = os.path.join('../data', self.args.city, self.args.city+str(int(self.args.scale*100000))+str(self.args.time_size))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.h5path = os.path.join("E://data", self.args.city+".h5")
        print("parameter setting： \n", self.args)

    def write(self, f, train_num, val_num, is_train):
        write_or_add = 'w'
        train_or_val = 'train' if is_train else 'val'
        start = 0 if is_train else train_num
        num = train_num if is_train else val_num
        val_num = 0

        writer = open(os.path.join(self.save_path, '{}'.format(train_or_val)), write_or_add)

        for i in range(start, start + num):
            trip = np.array(f.get('trips/' + str(i + 1)))  # numpy n*2
            ts = np.array(f.get('timestamps/' + str(i + 1)), dtype=np.int32)  # numpy n*1

            if len(trip) > settings.max_len:
                trips = []
                tss = []
                while len(trip) >= settings.max_len:
                    rand_len = np.random.randint(settings.min_len, settings.max_len)
                    trips.append(trip[0:rand_len])
                    trip = trip[rand_len:]
                    tss.append(ts[0:rand_len])
                    ts = ts[rand_len:]
                if len(trip) >= settings.min_len:
                    trips.append(trip)
                    tss.append(ts)
            else:
                trips = [trip]
                tss = [ts]

            for tr, ts in zip(trips, tss):
                if settings.min_len <= len(tr) <= settings.max_len:
                    mapIDs = self.trip2mapIDs(tr, ts)
                    val_num += 1
                    src_seq = ' '.join([str(id) for id in mapIDs])
                    writer.writelines(src_seq)
                    writer.write('\n')
            if i % 10000 == 0:
                print("{}schedule：{}/{} ,{}".format(train_or_val, i-start, num, time.ctime()))
        writer.close()
        return val_num

    # Divide the training set and test set and write the file train.src/ train.trg/ val.trg/ val.src
    def output_train_validate(self, train_ratio=settings.train_ratio, is_samll=True):
        f = h5py.File(self.h5path, 'r')
        trj_nums = f.attrs['num'] if not is_samll else 120000
        train_num = int(train_ratio*trj_nums)
        val_num = trj_nums - train_num
        print("Theoretically should generate the number of training set, test set trajectories", train_num, val_num)
        train_num = self.write(f, train_num, val_num, True)
        val_num = self.write(f, train_num, val_num, False)
        print("Limit the length of the training set, the number of test set trajectories", train_num, val_num)
        f.close()
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
        lon = self.args.lons[0]+x_offset*self.args.scale
        lat = self.args.lats[0]+y_offset*self.args.scale
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
        lon,lat = self.xyoffset2lonlat(x_offset, y_offset)
        return lon, lat
    
    ''' space_cell_id+t --> map_id  116,10->1796'''
    def spaceId2mapId(self, space_id, t):
        return int(space_id + t*self.args.space_cell_size)
    
    ''' map_id -->space_cell_id  1796-> 116'''
    def mapId2spaceId(self, map_id):
        return int(map_id % self.args.space_cell_size)
    
    def trip2mapIDs(self, trip, ts):
        map_ids = []
        for (lon, lat), t in zip(trip, ts):
            space_id = self.gps2spaceId(lon, lat)
            t = int(t) // self.args.time_span
            map_id = int(self.spaceId2mapId(space_id, t))
            map_ids.append(map_id)
        return list(map_ids)

    
if __name__ == "__main__":
    r = Region()
    r.output_train_validate(is_samll=True)
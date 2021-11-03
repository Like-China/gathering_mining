# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 17:46:09 2021

@author: likem
"""
import os
import numpy as np
import h5py
import warnings
warnings.filterwarnings("ignore")
import time
import constants
import argparse


'''
    设置空间cell和时间cell划分的参数
'''
def setRegionArgs():
        cityname = constants.cityname
        scale = constants.scale
        time_size = constants.time_size
    
        parser = argparse.ArgumentParser(description="Region.py")
        parser.add_argument("-cityname", default= cityname, help="城市名")
        if cityname == "beijing":
            lons_range, lats_range = constants.lons_range_bj,constants.lats_range_bj
        else:
            lons_range, lats_range = constants.lons_range_pt,constants.lats_range_pt
            
        '''获取在当前scale设定下，空间的划分'''
        minx, miny = 0,0
        maxx, maxy = (lons_range[1]-lons_range[0])//scale, (lats_range[1]-lats_range[0])//scale
        
        ''' 空间划分参数 '''
        parser.add_argument("-lons", default= lons_range, help="经度范围")
        parser.add_argument("-lats", default= lats_range, help="纬度范围")
        parser.add_argument("-scale", default= scale, help="空间单元格大小")
        parser.add_argument("-minx", type = int, default = minx, help="最小横坐标空间编号")
        parser.add_argument("-maxx", type = int, default = maxx, help="最大横坐标空间编号")
        parser.add_argument("-miny", type = int, default = miny, help="最小纵坐标空间编号")
        parser.add_argument("-maxy", type = int, default = maxy, help="最大纵坐标空间编号")
        parser.add_argument("-numx", type = int, default = maxx, help="空间上横块数")
        parser.add_argument("-numy", type = int, default = maxy, help="空间上纵块数")
        parser.add_argument("-space_cell_size", type = int, default = maxx*maxy, help="空间cell数")
        
        ''' 时间划分参数 '''
        parser.add_argument("-time_size", type = int, default = time_size, help="一天分为的时间段数目")
        parser.add_argument("-time_span", type = int, default = 86400 // time_size, help="每个时间段长度")
        
        
        ''' 时空格子参数 '''
        parser.add_argument("-start", type = int, default = constants.START, help="vocal word编码从4开始编号，0，1，2，3有特殊作用")
        parser.add_argument("-map_cell_size", type = int, default = maxx*maxy*time_size, help="时空格子数目（x,y,t)三维")
        
        args = parser.parse_args()
        return args


'''
    划分空间格子和时间格子
    对于每一个原始的（x,y,t)轨迹点，将其转化为对应的时空格子编码
    输入：
    args  设置的训练参数
    输出存储：
    ./data/cityname/cityname_regionScale_timeScale/train.src
    ./data/cityname/cityname_regionScale_timeScale/val.src
    
'''  
class Region:
    
    def __init__(self):
        self.args = setRegionArgs()
        
        if constants.cityname == 'beijing':
            self.dropping_rates = [0,0.2,0.3,0.4,0.5,0.6]
        else:
            self.dropping_rates = [0]
        
        '''
            结果数据文件存放路径 ,存储src的文件路径
            ./data/porto/..
        '''
        self.save_path = os.path.join(os.getcwd(),'data',self.args.cityname, self.args.cityname+str(int(self.args.scale*100000))+str(self.args.time_size))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        ''' 
            读取轨迹 h5文件路径 
            ./data/porto.h5
        
        '''
        self.h5path = os.path.join(os.getcwd(),'data', self.args.cityname+".h5")
        print("参数设置：")
        print(self.args)
        
    
    ''' 
        划分训练集和测试集并写入文件 train.src/ train.trg/ val.trg/ val.src 
    '''
    def createTrainVal(self, train_ratio=constants.train_ratio, is_samll = True):
        
        val_train_num = 0
        val_test_num = 0
        # 划分训练集和测试集
        f = h5py.File(self.h5path,'r')
        trj_nums = f.attrs['num'] if is_samll==False else 100000
        train_nums = int(train_ratio*trj_nums)
        val_nums = int((1-train_ratio)*trj_nums)
        
        # print('训练集数目：{}'.format(train_nums*len(self.dropping_rates)))
        # print('测试集数目：{}'.format(val_nums*len(self.dropping_rates)))
        write_or_add = 'w'
        train_src_writer = open(os.path.join(self.save_path,'train.src'), write_or_add)
        val_src_writer = open(os.path.join(self.save_path,'val.src'),  write_or_add)
        
        ''' 
            生成训练样本文件 
        '''
        # for i in tqdm(range(train_nums),desc='train split'):
        for i in range(train_nums):
            ''' ready for write'''
            trip = np.array(f.get('trips/'+str(i+1))) # numpy n*2
            ts = np.array(f.get('timestamps/'+str(i+1))) # numpy n*1
            if len(trip)>=100 or len(trip)< 20:
                # trips = np.array_split(trip, len(trip)//50)
                # tss = np.array_split(ts, len(trip)//50)
                continue
            else:
                trips = np.array_split(trip, 1)
                tss = np.array_split(ts, 1)
                for ii in range(len(trips)):
                    tr = trips[ii]
                    ts = tss[ii]
                    # flag =  0
                    # for coor in tr:
                    #     if(coor[0]<=constants.lons_range_pt[0] or coor[0]>=constants.lons_range_pt[1]):
                    #         flag = 1
                    #         break
                    #     if(coor[1]<=constants.lats_range_pt[0] or coor[0]>=constants.lats_range_pt[1]):
                    #         flag = 1
                    #         break
                    # if (flag == 1):
                    #     break
                    mapIDs = self.trip2mapIDs(tr,ts) # list 
                    multi_mapIDs = self.subsample(mapIDs)
                    for seq in multi_mapIDs:
                         if len(seq)>=20 and len(seq)<=100:
                            if min(seq)<0:
                                break
                            val_train_num = val_train_num+1
                            src_seq = ' '.join([str(id) for id in seq])
                           
                            train_src_writer.writelines(src_seq)
                            train_src_writer.write('\n')
           
            if i % 10000 == 0:
                print("训练集生成进度：{}/{} ,{} \n".format(i, train_nums, time.ctime()))
            ''' write src'''
            
        train_src_writer.close()
        
        
        ''' 
            生成测试样本文件 
        '''
        for i in range(val_nums):
            ''' ready for write'''
            trip = np.array(f.get('trips/'+str(i+1+train_nums))) # numpy n*2
            ts = np.array(f.get('timestamps/'+str(i+1+train_nums))) # numpy n*1
            if len(trip)>=100 or len(trip)< 20:
                continue
                # trips = np.array_split(trip, len(trip)//50)
                # tss = np.array_split(ts, len(trip)//50)
            else:
                trips = np.array_split(trip, 1)
                tss = np.array_split(ts, 1)
                for ii in range(len(trips)):
                    tr = trips[ii]
                    ts = tss[ii]
                    mapIDs = self.trip2mapIDs(tr,ts) # list 
                    multi_mapIDs = self.subsample(mapIDs)
                    for seq in multi_mapIDs:
                         if min(seq)<0:
                             break
                         if len(seq)>=20 and len(seq)<=100:
                            src_seq = ' '.join([str(id) for id in seq])
                            val_test_num = val_test_num +1
                            val_src_writer.writelines(src_seq)
                            val_src_writer.write('\n')
            if i % 10000 == 0:
                print("测试集生成进度：{}/{}, {} \n".format(i, val_nums, time.ctime()))
        val_src_writer.close()
        print('有效轨迹数:')
        print(val_test_num)
        print(val_train_num)
        print(val_train_num+val_test_num)
        f.close()

    '''
    ****************************************************************************************************************************************************
    一系列的转化函数
    ****************************************************************************************************************************************************
    '''
    
    '''经纬度转换为米为单位, 映射到平面图上 (116.3, 40.0)->(4,8)'''
    def lonlat2xyoffset(self,lon, lat):
        xoffset = round((lon - self.args.lons[0]) / self.args.scale)
        yoffset = round((lat - self.args.lats[0]) / self.args.scale)
        return xoffset, yoffset


    ''' 米单位转换为经纬度  (4,8)-> (116.3, 40.0)'''
    def xyoffset2lonlat(self, xoffset, yoffset):
        lon = self.args.lons[0]+xoffset*self.args.scale
        lat = self.args.lats[0]+yoffset*self.args.scale
        return lon,lat
    
    
    ''' (xoffset,yoffset) -> space_cell_id  (4,8)->116'''
    def offset2spaceId(self, xoffset, yoffset):
        return yoffset * self.args.numx + xoffset
    
    
    ''' space_cell_id -->(x,y) 116->(4.8)'''
    def spaceId2offset(self, space_cell_id):
        yoffset = space_cell_id // self.args.numx
        xoffset = space_cell_id % self.args.numx
        return xoffset, yoffset
    
    
    ''' gps--> space_cell_id  116.3,40->116'''
    def gps2spaceId(self, lon, lat):
        xoffset, yoffset = self.lonlat2xyoffset(lon, lat)
        space_cell_id = self.offset2spaceId(xoffset, yoffset)
        return space_cell_id
    
    
    '''space_cell_id -->gps 116->116.3,40'''
    def spaceId2gps(self, space_cell_id):
        xoffset, yoffset = self.spaceId2offset(space_cell_id)
        lon,lat = self.xyoffset2lonlat(xoffset,yoffset)
        return lon,lat
    
    ''' space_cell_id+t --> map_id  116,10->1796'''
    def spaceId2mapId(self, space_id, t):
        return space_id + t*self.args.space_cell_size
    
    ''' map_id -->space_cell_id  1796-> 116'''
    def mapId2spaceId(self, map_id):
        return map_id % self.args.space_cell_size
    
    
    ''' 减少迭代次数的trip2ids '''
    def trip2mapIDs(self, trip, ts):
        map_ids = []
        for (lon,lat),t in zip(trip,ts):
            space_id = self.gps2spaceId(lon, lat)
            t = int(t) // self.args.time_span
            map_id = int(self.spaceId2mapId(space_id,t))
            map_ids.append(map_id)
        return list(map_ids)
    
    ''' 对每一条trajectory token sequence, 通过下采样获得更多可用的轨迹 '''
    def subsample(self, mapId):
        res = []
        for r in self.dropping_rates:
            randx = np.random.rand(len(mapId))>r
            res.append(np.array(mapId)[randx].tolist())
        return res
    
if __name__ == "__main__":
    r = Region()
    r.createTrainVal(is_samll=True)

        
    
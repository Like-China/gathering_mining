import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import h5py
import warnings
import time
warnings.filterwarnings("ignore")

import constants

'''
    读取一定数量第taxi文本
    生成满足条件的轨迹文本 h5
    对北京轨迹数据集做处理，将其处理成为porto.h5一样的格式
'''
class Processor:
    
    
    def __init__(self, read_txt_size, each_read):
        # 经度范围
        self.longtitude_range = constants.lons_range_bj
        # 维度范围
        self.latitude_range = constants.lats_range_bj
        self.current_read = 0
        self.each_read = each_read
        # 读取的taxt txt文本数,5000中大概能够提取出10K
        self.read_txt_size = read_txt_size
        # 重写src/trg 还是尾部添加
        self.min_length = 30
        self.max_length = 100
        self.min_duration = 7200
        self.max_duration = 40000
        # taxi 文本文件路径
        self.file_dir = os.path.join(os.getcwd(),"data","taxi_log_2008_by_id")
        # 当前写了多少个trips到h5文件中
        self.curent_nums = 0
        self.h5_filepath = os.path.join(os.getcwd(),"data","beijing.h5")
        
    
    # 一批一批地读入taxi文本并写出到h5文件
    def go(self): 
        f = h5py.File(self.h5_filepath,'w')
        for ii in tqdm(range(self.read_txt_size//self.each_read)):
            # 读取固定条文本轨迹，存储到self.all_raw_trjs中
            self.read_files()
            # 处理数据，生成 tajs_with_time, tajs_with_ts
            self.address_data()
            # 写出数据
            self.output_src_tgr(f)
            ## 读取下一批轨迹
            self.current_read += self.each_read
        f.close()
        print("\n共写出轨迹："+str(self.curent_nums))
    
    
    # 写出一批轨迹
    def output_src_tgr(self,f):
        
        for each_taj in self.tajs_with_ts:
            if(len(each_taj)>self.max_length or len(each_taj)<self.min_length):
                continue
            # 转换为array形式
            each_taj = np.array(each_taj)
            xys = each_taj[:,0:2]
            times = [each[0] for each in each_taj[:,2:3]]
            
            #检测轨迹维持的时间长度,一条轨迹时间持续一小时以上才记录
            if(times[-1]-times[0]>self.min_duration and times[-1]-times[0]<self.max_duration):
                self.curent_nums += 1
                f["trips/"+str(self.curent_nums)] = xys
                f["timestamps/"+str(self.curent_nums)]=times
        f.attrs['num'] = self.curent_nums
        
    '''
        从一定数目的taxi txt轨迹文本中读取一批轨迹信息, 存储到 self.all_raw_trjs中
    '''
    def read_files(self):
        self.trjs_raw = []
        # 获取所有taxi轨迹文本
        all_file_list = os.listdir(self.file_dir)
        all_file_list.sort(key=lambda x: int(x[:-4]))
        all_file_list = all_file_list[self.current_read:self.current_read+self.each_read]
        all_data = pd.DataFrame()
        # 读取一定数目的 轨迹文本 获取信息
        for file in all_file_list:
            single_data = pd.read_csv(os.path.join(self.file_dir, file), names=['id', 'times', 'longitude', 'latitude'], header=None)
            ## 如果该 taxi文本记录的轨迹信息太少，则不记录该轨迹信息
            if(len(single_data)) < 400:
               continue
            all_data = all_data.append(single_data)
    
        # 过滤不在限制经纬度区域范围内的数据
        all_data = all_data[self.longtitude_range[0] < all_data.longitude]
        all_data = all_data[all_data.longitude < self.longtitude_range[1]]
        all_data = all_data[self.latitude_range[0] < all_data.latitude]
        all_data = all_data[all_data.latitude < self.latitude_range[1]]
        
        str_times = list(all_data['times'])
        longitudes = list(all_data['longitude'])
        latitudes = list(all_data['latitude'])
        
        location = list(zip(longitudes, latitudes))
        i = 0
        while i < len(location):
            cur_date = str_times[i]
            single_taj_with_time = []
            '''cur_date[:10] == str_times[i][:10] 控制在同一天'''
            while i < len(location) and cur_date[:10] == str_times[i][:10]:
                single_taj_with_time.append((str_times[i][11:], location[i]))
                i += 1
            '''在这里也控制下最小长度'''
            if(len(single_taj_with_time)>self.min_length):
                self.trjs_raw.append(single_taj_with_time)
   
    
    # 处理一批轨迹
    def address_data(self):
        ## [经纬度坐标，时间]
        self.tajs_with_time = [] 
        ## [经纬度坐标]
        self.tajs_with_ts = []
        # taj是一条轨迹 [('15:36:08', (116.51172, 39.92123)), ('15:46:08', (116.51135, 39.938829999999996))] 
        for nn in range(len(self.trjs_raw)):
            # # 单条轨迹不能超过最大长度 max_length，如果大于需要划分
            self.split_taj(self.trjs_raw[nn])
            
                
    # 如果轨迹太长，则划分轨迹
    def split_taj(self, taj):
        # taj = [('13:33:52', (116.36421999999999, 39.887809999999995))]
        tajs = []
        while(len(taj) >= 100):
            # 将轨迹按min-max轨迹长度生成一个随机长度的轨迹
            rand_len = np.random.randint(20,100)
            tajs.append(taj[0:rand_len])
            taj = taj[rand_len:]
        if(len(taj)>20):
            tajs.append(taj)
        else:
            tajs= [taj]
            
        for ii in range(len(tajs)):
            t = tajs[ii]
            single_taj_with_time = []
            single_taj = []
            # 记录前一个时间戳，防止记录过于频繁的轨迹
            last_timestamp = 0
            flag = True
            for jj in range(len(t)):
                time = t[jj][0]
                # 获取时间的时分秒
                h,m,s = [int(i) for i in time.split(":")]
                # 得到以一天开头为零点的时间戳
                timestamp = h*3600+m*60+s
                # 过滤记录频繁的轨迹
                if(timestamp-last_timestamp < 100):
                    flag = False
                    break
                longtitude,latitude = t[jj][1][0],t[jj][1][1]
                single_taj.append([longtitude,latitude,timestamp])
                single_taj_with_time.append([longtitude,latitude,time])
            if(flag):
                self.tajs_with_ts.append(single_taj) 
                self.tajs_with_time.append(single_taj_with_time)
        

if __name__ == "__main__":
    print(time.ctime())
    P = Processor(10200,200)
    P.go()
    print(time.ctime())
   
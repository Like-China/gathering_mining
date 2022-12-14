import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import h5py
import warnings
import time
import constants
warnings.filterwarnings("ignore")


'''
    Read a certain number of Taxi texts
    Generate trace text that meets the conditions h5 format
    The Beijing trajectory data set is processed into a format similar to porto.h5
'''
class Processor:

    def __init__(self, read_txt_size, each_read):
        self.longtitude_range = constants.lons_range_bj
        self.latitude_range = constants.lats_range_bj
        self.current_read = 0
        self.each_read = each_read
        self.read_txt_size = read_txt_size
        self.min_length = 30
        self.max_length = 100
        self.min_duration = 7200
        self.max_duration = 40000
        self.file_dir = os.path.join(os.getcwd(),"data","taxi_log_2008_by_id")
        self.curent_nums = 0
        self.h5_filepath = os.path.join(os.getcwd(),"data","beijing.h5")

    def go(self): 
        f = h5py.File(self.h5_filepath, 'w')
        for ii in tqdm(range(self.read_txt_size//self.each_read)):
            self.read_files()
            self.address_data()
            self.output_src_tgr(f)
            self.current_read += self.each_read
        f.close()
        print("\nThe total number of trajectories written："+str(self.curent_nums))

    def output_src_tgr(self,f):
        
        for each_taj in self.tajs_with_ts:
            if(len(each_taj)>self.max_length or len(each_taj)<self.min_length):
                continue
            each_taj = np.array(each_taj)
            xys = each_taj[:,0:2]
            times = [each[0] for each in each_taj[:,2:3]]
            
            # The length of time that the trajectory is maintained is recorded only when the time of a trajectory lasts more than one hour
            if(times[-1]-times[0]>self.min_duration and times[-1]-times[0]<self.max_duration):
                self.curent_nums += 1
                f["trips/"+str(self.curent_nums)] = xys
                f["timestamps/"+str(self.curent_nums)]=times
        f.attrs['num'] = self.curent_nums
        
    '''
        Read a batch of trajectories information from a certain number of Taxi txt trajectory texts, store in self.all_raw_trjs
    '''
    def read_files(self):
        self.trjs_raw = []
        all_file_list = os.listdir(self.file_dir)
        all_file_list.sort(key=lambda x: int(x[:-4]))
        all_file_list = all_file_list[self.current_read:self.current_read+self.each_read]
        all_data = pd.DataFrame()
        for file in all_file_list:
            single_data = pd.read_csv(os.path.join(self.file_dir, file), names=['id', 'times', 'longitude', 'latitude'], header=None)
            if(len(single_data)) < 400:
               continue
            all_data = all_data.append(single_data)
    
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
            '''cur_date[:10] == str_times[i][:10] Control on the same day'''
            while i < len(location) and cur_date[:10] == str_times[i][:10]:
                single_taj_with_time.append((str_times[i][11:], location[i]))
                i += 1
            '''Here also controls the minimum length'''
            if(len(single_taj_with_time)>self.min_length):
                self.trjs_raw.append(single_taj_with_time)
   
    
    def address_data(self):
        self.tajs_with_time = [] 
        self.tajs_with_ts = []
        for nn in range(len(self.trjs_raw)):
            self.split_taj(self.trjs_raw[nn])
            
                
    def split_taj(self, taj):
        # taj = [('13:33:52', (116.36421999999999, 39.887809999999995))]
        tajs = []
        while(len(taj) >= 100):
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
            last_timestamp = 0
            flag = True
            for jj in range(len(t)):
                time = t[jj][0]
                h,m,s = [int(i) for i in time.split(":")]
                timestamp = h*3600+m*60+s
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
   
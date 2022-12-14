import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import h5py
import warnings
import time
warnings.filterwarnings("ignore")


class Processor:
    """
    Read a certain number of Taxi texts and generate track text h5 that meets the conditions

    trips       trips/0 -> [[longitude, latitude], ...]

    timestamps     timestamps/0 -> [ts, ts,...]

    f.attrs['num']  ：The total number of valid tracks recorded
    """
    def __init__(self, read_txt_size, each_read):
        self.longtitude_range = [116.25, 116.55]
        self.latitude_range = [39.83, 40.03]
        self.current_read = 0
        self.each_read = each_read
        self.read_txt_size = read_txt_size
        self.min_length, self.max_length = 20, 100
        self.file_dir = os.path.join("F://data/taxi_log_2008_by_id")
        self.valid_trip_nums = 0
        self.h5_filepath = os.path.join("F://data/beijing.h5")
        self.freq_interval = 10

    def go(self):
        """
        Read the taxi text in batches and write it into the h5 file
        :return:
        """
        f = h5py.File(self.h5_filepath, 'w')
        for ii in tqdm(range(self.read_txt_size//self.each_read)):
            batch_trjs = self.read_files()
            self.address_data(batch_trjs)
            self.output_src_tgr(f)
            self.current_read += self.each_read
        f.close()
        print("\n The total number of writed trajectories："+str(self.valid_trip_nums))

    def read_files(self):
        """
        Read a batch of trajectory information from a certain number of taxi txt trajectory text, store it in raw_trjs and return it
        :return: trjs_raw is a fixed number of txt files, Within the latitude and longitude range, each record is a track within the same day, and meets the minimum length limit of the track
        """
        trjs_raw = []
        all_file_list = os.listdir(self.file_dir)
        all_file_list.sort(key=lambda x: int(x[:-4]))
        all_file_list = all_file_list[self.current_read:self.current_read + self.each_read]
        all_data = pd.DataFrame()
        for file in all_file_list:
            single_data = pd.read_csv(os.path.join(self.file_dir, file), ['id', 'times', 'longitude', 'latitude'],
                                      header=None)
            if (len(single_data)) < 100: continue
            all_data = all_data.append(single_data)

        all_data = all_data[self.longtitude_range[0] <= all_data.longitude]
        all_data = all_data[all_data.longitude <= self.longtitude_range[1]]
        all_data = all_data[self.latitude_range[0] <= all_data.latitude]
        all_data = all_data[all_data.latitude <= self.latitude_range[1]]

        str_times = list(all_data['times'])
        longitudes = list(all_data['longitude'])
        latitudes = list(all_data['latitude'])
        location = list(zip(longitudes, latitudes))

        i = 0
        while i < len(location):
            cur_date = str_times[i]
            single_taj_with_time = []
            while i < len(location) and cur_date[:10] == str_times[i][:10]:
                single_taj_with_time.append((str_times[i][11:], location[i]))
                i += 1
            if len(single_taj_with_time) >= self.min_length:
                trjs_raw.append(single_taj_with_time)
        return trjs_raw

    def address_data(self, batch_trjs):
        """
        Process a batch of tracks, and divide tracks if the tracks are too long
        """
        # [Latitude and longitude coordinates, time]
        self.tajs_with_time = []
        # [Latitude and longitude coordinates, timestamp of the transformation]
        self.tajs_with_ts = []
        for taj in batch_trjs:
            # A single track cannot exceed the maximum length(max_length). If the length is larger than max_length, it needs to be divided
            # taj = [('13:33:52', (116.36421999999999, 39.887809999999995))]
            tajs = []
            while len(taj) >= self.max_length:
                # Generate a trajectory of random length according to the length of min-max trajectory
                rand_len = np.random.randint(self.min_length, self.max_length)
                tajs.append(taj[0:rand_len])
                taj = taj[rand_len:]
            if len(taj) >= self.min_length:
                tajs.append(taj)

            for ii in range(len(tajs)):
                # t = [('13:33:52', (116.36421999999999, 39.887809999999995))]
                t = tajs[ii]
                single_taj_with_time = []
                single_taj_with_ts = []
                # Record the previous timestamp to prevent tracks from being recorded too frequently
                last_timestamp = 0
                flag = True
                for jj in range(len(t)):
                    time = t[jj][0]
                    h, m, s = [int(i) for i in time.split(":")]
                    timestamp = h * 3600 + m * 60 + s
                    # Filter records frequent trajectory
                    if timestamp - last_timestamp <= self.freq_interval:
                        flag = False
                        break
                    longtitude, latitude = t[jj][1][0], t[jj][1][1]
                    single_taj_with_ts.append([longtitude, latitude, timestamp])
                    single_taj_with_time.append([longtitude, latitude, time])
                # If there is no more frequent sampling interval in the trajectory, then record
                if flag:
                    self.tajs_with_ts.append(single_taj_with_ts)
                    self.tajs_with_time.append(single_taj_with_time)

    def output_src_tgr(self, f):
        """
        Write a batch of trajectories
        :param f: Target h5 file
        """
        for each_taj in self.tajs_with_ts:
            if self.min_length <= len(each_taj) <= self.max_length:
                self.valid_trip_nums += 1
                each_taj = np.array(each_taj)
                locations = each_taj[:, 0:2]
                times = [each[0] for each in each_taj[:, 2:3]]
                f["trips/"+str(self.valid_trip_nums)] = locations
                f["timestamps/"+str(self.valid_trip_nums)] = times
        f.attrs['num'] = self.valid_trip_nums


class ProcessorTester:
    """
    Check whether the generated H5 file is reasonable and whether there are errors
    """
    def __init__(self):
        self.h5_filepath = os.path.join(os.getcwd(), "data", "beijing.h5")
        f = h5py.File(self.h5_filepath, 'r')
        self.checked_nums = min(100000, f.attrs['num'])

    def observe(self):
        with h5py.File(self.h5_filepath, 'r') as f:
            for ii in tqdm(range(self.checked_nums)):
                trip = np.array(f.get('trips/' + str(ii + 1)))
                ts = np.array(f.get('timestamps/' + str(ii + 1)))
                print(trip)
                print(ts)

    def check_lens(self):
        """
        Check whether the track length is reasonable
        """
        with h5py.File(self.h5_filepath, 'r') as f:
            # print(f.keys())
            lens = []
            for ii in tqdm(range(self.checked_nums)):
                trip = np.array(f.get('trips/' + str(ii + 1)))
                ts = np.array(f.get('timestamps/' + str(ii + 1)))
                lens.append(len(ts))
                assert 20 <= len(ts) <= 100, "The length exceeds the range limit"
                assert len(trip) == len(ts), "locations The length exceeds the range limit!!"


if __name__ == "__main__":

    t1 = time.time()
    P = Processor(10200, 50)
    P.go()
    print("The time of get the h5 files ：", time.time()-t1)



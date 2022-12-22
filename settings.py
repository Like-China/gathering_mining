import argparse
import os


# Constant parameters
min_len, max_len = 20, 50
# beijing
lons_range_bj = [116.25, 116.55]
lats_range_bj = [39.83, 40.03]
# porto
lons_range_pt = [-8.735, -8.156]
lats_range_pt = [40.953,  41.307]
process_num = 40
# experimental number
n = 20

# variable parameters
city = "beijing"
# default settings
if city == "beijing":
    num = 6000
    nums = [2000, 4000, 6000, 8000, 10000]
else:
    num = 20000
    nums = [20000, 40000, 60000, 80000, 100000]
dist_error, time_error = 200, 200
min_lifetime = 4
time_size = 40
scale = 0.002
ep = 0.5
min_group_trj_num = 4
# vary range

# dist_errors = [100,200,300,400,500]
# scales = [0.001,0.002,0.003,0.004,0.005]
# time_errors = [100,200,300,400,500]
# time_sizes = [50,100,150,200,250]
min_life_times = [2,3,4,5,6]
eps = [0.42,0.44,0.46,0.48,0.50]
min_group_trj_nums = [2,3,4,5,6]




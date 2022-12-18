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
process_num = 10

# variable parameters
# 最长公共子序列误差设置
dist_error = 150
time_error = 60465 // 400
city = "porto"
scale = 0.002 #
time_size = 400 # timestamp range [1 60465]
min_lifetime = 3
min_group_trj_nums = 3
# SCAN参数
ep = 0.4 # +
#




## 存储一些通用的参数设置

PAD = 0
BOS = 1
EOS = 2
UNK = 3 # 低频词编码
START = 4 # 从4开始编码

PAD_WORD = '<blank>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
UNK_WORD = '<unk>'

# beijing
lons_range_bj = [116.25,116.55]
lats_range_bj = [39.83,40.03]
# porto
lons_range_pt = [-8.735,-8.156]
lats_range_pt = [40.953, 41.307]
# 训练集比例
train_ratio = 0.8


# 常规参数
cityname = "porto"
scale = 0.015
time_size = 450
min_lifetime = 5
min_group_trj_nums = 10
each_num = 8 # 每组选择多少人计算t_loss
# 聚类参数
c_method = 1
n = 30 #聚类数目
# DBSCAN参数
eps = 0.11
mt = 10
# 训练集测试集每次取得数目
train_batch = 5000
val_batch = 10000





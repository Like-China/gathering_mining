import numpy as np
import torch
import constants
from collections import namedtuple
from tqdm import tqdm
from funcy import merge
import time

def argsort(seq):
    """
    sort by length in reverse order
    ---
    seq (list[array[int32]])
    将一个句子集合按照序列长度长大到小排列,返回id集合 
    如src=[[1,2,3],[3,4,5,6],[2,3,4,56,3]] ，返回2，1，0
    """
    return [x for x,y in sorted(enumerate(seq),
                                key = lambda x: len(x[1]),
                                reverse=True)]


def pad_array(a, max_length, PAD=constants.PAD):
    """
    a (array[int32])
    单条轨迹补零操作 将长度补齐为 该批轨迹的最大长度
    [1,2,3] -> [1,2,3,0,0,..]
    """
    return np.concatenate((a, [PAD]*(max_length - len(a))))


def pad_arrays(a):
    """
    a array(array[int32])
    多条轨迹补零操作 每条轨迹的长度补0为和该batch最长轨迹的长度相同
    """
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int)
    return torch.LongTensor(a)


def pad_arrays_pair(src):
    """
    Input:
    src (list[array[int32]])
    ---
    Output:
    src (seq_len1, batch)
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    
    1. 对轨迹补零操作，使得所有轨迹的长度都一样长
    2. 对轨迹长度从大到小进行排序
    3. 返回TD类，其中轨迹点列表进行了转置操作，每列代表一个轨迹
    4. 返回形式 ['src', 'lengths', 'invp']
    """
    TD = namedtuple('TD', ['src', 'lengths', 'invp'])

    idx = argsort(src)
    src = list(np.array(src)[idx])

    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    
    invp = torch.LongTensor(invpermute(idx))
    # (batch, seq_len) => (seq_len, batch)
    return TD(src=src.t().contiguous(), lengths=lengths.view(1, -1), invp=invp)
   

def invpermute(p):
    """
    inverse permutation
    输入p,返回p的每个位置的值的索引invp
    idx = [5, 7, 8, 9, 6, 1, 2, 0, 3, 4]
    invp(idx) = [7, 5, 6, 8, 9, 0, 4, 1, 2, 3]  
    invp[p[i]] = i 如p中有个数是45，我现在想知道45在p的第几个位置，那么invp[45]会告诉我们答案
    invp[i] = p.index(i)
    """
    p = np.asarray(p)
    invp = np.empty_like(p)
    for i in range(p.size):
        invp[p[i]] = i
    return invp


def random_subseq(a, rate):
    """
    以一定概率去除a[1:-2]中的点
    Dropping some points between a[3:-2] randomly according to rate.

    Input:
    a (array[int])
    rate (float)
    """
    idx = np.random.rand(len(a)) < rate
    idx[0], idx[-1] = True, True
    return a[idx]


class DataLoader():
    """
    训练集无序加载数据，测试集合有序加载数据
    """
    def __init__(self, srcfile, batch):
        self.srcfile = srcfile
        self.batch = batch
        # 记录最大ID
        self.maxID = 0
        self.minID = 100
    '''
    加载所有轨迹
    trjs存在，轨迹直接作为参数加载，返回list[list]
    trjs不存在，从指定的文件夹读取,返回list[array]
    '''
    def load(self, max_num_line=0):
        self.all_trjs = []
        ''' 将不同长度的轨迹丢入到 限制src长度和 trg长度的 篮子里面 '''
        srcstream = open(self.srcfile, 'r')
        num_line = 0
        # with tqdm(total=max_num_line, desc='Reading Traj', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
        for s in srcstream:
            s = [int(x)+constants.START for x in s.split()]
            if len(s)>0:
                # self.all_trjs[num_line] = np.array(s, dtype=np.int32)
                self.all_trjs.append(s)
                num_line += 1
                if max(s)>self.maxID:
                    self.maxID = max(s)
                if min(s)<self.minID:
                    self.minID = min(s)
            # pbar.update(1)
            if num_line >= max_num_line and max_num_line > 0: break
        srcstream.close()
        self.size = len(self.all_trjs)
        
        
        
    '''
    求两个有序序列的公共序列
    '''
    def commom_vals(self, m, n):
        minLen = min(len(m),len(n))
        if(m[0]>=n[-1] or n[0]>=m[-1]):
            return []
        i = 0
        j = 0
        cvals = []
        while (j < minLen and i < minLen):
            if (m[i] == n[j]):
                cvals.append(m[i]);
                i += 1
                j += 1
            elif (m[i] < n[j]):
                i += 1
            else:
                j += 1
        return cvals;

    
    def get_pairs(self, min_group_trj_nums, min_lifetime):
        t1 = time.time()
        all_common_pairs = []
        ''' 
        记录所有的group
        '''
        trj_num = len(self.trjs)
        for i in range(trj_num):
            common_pairs = []
            for j in range(i+1,trj_num):
                ''' 对已经形成group的轨迹，不再去组成新的group '''
                cvals = self.commom_vals(self.trjs[i],self.trjs[j])
                if(len(cvals)>=min_lifetime):
                    common_pairs.append([i,j,cvals])
            all_common_pairs.append(common_pairs)
        if len(all_common_pairs)>0:
            all_common_pairs = merge(*all_common_pairs)
        print('得到所有组用时:'+str(time.time()-t1))
        return all_common_pairs
        
    ''' 
    计算得到全部的group
    返回所有的co-movent pair
    '''
    def get_groups(self, min_lifetime, min_group_trj_nums):
        all_common_pairs = []
        ''' 
        记录所有的group
        '''
        self.cover_num = 0
        self.all_groups = []
        trj_num = len(self.trjs)
        all_grouped_sets = set()
        for i in tqdm(range(trj_num)):
            if i in all_grouped_sets:
                continue;
            common_pairs = []
            for j in range(i+1,trj_num):
                ''' 对已经形成group的轨迹，不再去组成新的group '''
                if j in all_grouped_sets:
                    continue;
                cvals = self.commom_vals(self.trjs[i],self.trjs[j])
                if(len(cvals)>=min_lifetime):
                    common_pairs.append([i,j,cvals])
            if(len(common_pairs)>=min_group_trj_nums):
                all_common_pairs.append(common_pairs)
                
                for ii in range(max(len(common_pairs)-min_group_trj_nums,1)):
                    group = [common_pairs[ii][0],common_pairs[ii][1]]
                    trj1 = common_pairs[ii][2]
                    for jj in range(ii+1,len(common_pairs)):
                        trj2 = common_pairs[jj][2]
                        cvals = self.commom_vals(trj1,trj2)
                        if(len(cvals)>=min_lifetime):
                            group.append(common_pairs[jj][1])
                    if(len(group)>=min_group_trj_nums):
                        self.all_groups.append(group)
                        self.cover_num += (len(group)-1)*(len(group)-1)//2
                        all_grouped_sets.update(set(group))
                        break
        '''
        输出每条轨迹属于哪个组的映射，列表形式
        若不属于某个组，则赋值为-1
        '''
        self.trj2groupIDs = [-1]*trj_num
        for ii in range(len(self.all_groups)):
            for trj_id in self.all_groups[ii]:
                self.trj2groupIDs[trj_id] = ii
        
        # n1 = sum(np.array(self.trj2groupIDs) != -1)
        # n2 = sum(np.array(self.trj2groupIDs) == -1)
        # print('归组人数:{0}\n未归组人数：{1}\n组数：{2}'.format(n1, n2,len(self.all_groups)))
        if len(all_common_pairs)>0:
            all_common_pairs = merge(*all_common_pairs)
        return all_common_pairs
    """ 
    随机获取获取一批batch条轨迹用于训练
    计算该组轨迹产生的group
    each_nums 每次每组选取多少个轨迹
    """
    def getbatch_one(self, each_nums, is_val = False):
        if self.batch<self.size:
            rand_index = np.random.randint(0,self.size,self.batch)
        else:
            rand_index = np.random.randint(0,self.size,self.size)
        
        if is_val == False:
            self.trjs = np.array(self.all_trjs)[rand_index].tolist()
            self.get_groups(constants.min_lifetime, constants.min_group_trj_nums)
        else:
            self.trjs = self.all_trjs
            self.get_groups(constants.min_lifetime, constants.min_group_trj_nums)
        
        # self.selected_trj_ids = []
        # self.selected_trjs = []
        # for group in self.all_groups:
        #     for ii in range(each_nums):
        #         selected_id = random.choice(group)
        #         self.selected_trj_ids.append(selected_id)
        #         self.selected_trjs.append(self.all_trjs[selected_id])
        # self.selected_trjs_pap = pad_arrays_pair(self.selected_trjs)

    '''
        以一定概率去除一批batch个数轨迹中的点后生成三个轨迹集合a, p，n
        a,p，n均取自同一条轨迹，a,p采样的子轨迹overlap程度比a,n采样子轨迹的overlap程度更高， 即a,p更为相似
    '''
    def getbatch_discriminative_inner(self):
        a_src,p_src,n_src = [],[],[]

        selected_trj_ids = np.random.choice(len(self.all_trjs), self.batch).tolist()
        trgs = np.array(self.all_trjs)[selected_trj_ids].tolist()
        for i in range(len(trgs)):
            trg = np.array(trgs[i])
            if len(trg) < 5: continue
            a1, a3, a5 = 0, len(trg)//2, len(trg)
            a2, a4 = (a1 + a3)//2, (a3 + a5)//2
            rate = np.random.choice([0.3, 0.4, 0.6])
            if np.random.rand() > 0.5:
                a_src.append(random_subseq(trg[a1:a4], rate))
                p_src.append(random_subseq(trg[a2:a5], rate))
                n_src.append(random_subseq(trg[a3:a5], rate))
            else:
                a_src.append(random_subseq(trg[a2:a5], rate))
                p_src.append(random_subseq(trg[a1:a4], rate))
                n_src.append(random_subseq(trg[a1:a3], rate))
        
        a = pad_arrays_pair(a_src)
        p = pad_arrays_pair(p_src)
        n = pad_arrays_pair(n_src)
        return a, p, n
    
    '''
        以一定概率去除一批batch个数轨迹中的点后生成三个轨迹集合a, p，n
        a,p,n均取自同一条轨迹，p和n 与 a 的overlap程度一样，但是ap的共同元素个数更多
        a = trg[a2:a4]
        p = 公共元素更多的 trg[a1:a3], trg[a3:a5]
    '''
    def getbatch_discriminative_common(self):
        
        a_src,p_src,n_src = [],[],[]
        selected_trj_ids = np.random.choice(len(self.all_trjs), self.batch).tolist()
        trgs = np.array(self.all_trjs)[selected_trj_ids].tolist()
        for i in range(len(trgs)):
            trg = np.array(trgs[i])
            if len(trg) < 5: continue
            rate = np.random.choice([0.3, 0.4, 0.6])
            sub1 = random_subseq(trg, rate)
            sub2 = random_subseq(trg, rate)
            sub3 = random_subseq(trg, rate)
            # 计算交集个数
            common_num1 = list(set(sub1).intersection(set(sub2)))
            common_num2 = list(set(sub1).intersection(set(sub3)))
            
            # 令sub2 为 p
            if len(common_num1) < len(common_num2):
                sub2, sub3 = sub3, sub2
            
            a_src.append(sub1)
            p_src.append(sub2)
            n_src.append(sub3)
            
        a = pad_arrays_pair(a_src)
        p = pad_arrays_pair(p_src)
        n = pad_arrays_pair(n_src)
        return a, p, n

    
    ''' 计算轨迹所属的时间段 '''
    def period(self,trj):
        cityname = constants.cityname
        scale = constants.scale
        time_size = constants.time_size
        if cityname == "beijing":
            lons_range, lats_range = constants.lons_range_bj,constants.lats_range_bj
        else:
            lons_range, lats_range = constants.lons_range_pt,constants.lats_range_pt
                
        '''获取在当前scale设定下，空间的划分'''
        maxx, maxy = (lons_range[1]-lons_range[0])//scale, (lats_range[1]-lats_range[0])//scale 
        space_size = maxx*maxy 
        time1 =  trj[0] // space_size
        time1 = (24* time1) //time_size 
        time2 = trj[-1] // space_size
        time2 = (24* time2) //time_size 
        if time1>=6 and time1<=11 and time2>=6 and time2<=11:
            return 1
        elif time1>=17 and time1<=20 and time2>=17 and time2<=20:
            return 1
        elif time1>=10 and time1<=17 and time2>=10 and time2<=17:
            return 2
        else:
            return 3
        
    
    ''' 根据时间段加载数据'''
    def load_by_period(self): 
        self.peak_trjs = []
        self.work_trjs = []
        self.casual_trjs = []
        for trj in self.all_trjs:
            if self.period(trj) == 1:
                self.peak_trjs.append(trj)
            elif self.period(trj) == 2:
                self.work_trjs.append(trj)
            else:
                self.casual_trjs.append(trj)
        print("peak_num:{}, work_num:{},casual_num:{}".format(
            len(self.peak_trjs), len(self.work_trjs),len(self.casual_trjs)))
        return pad_arrays_keep_invp(self.peak_trjs),pad_arrays_keep_invp(self.work_trjs), \
    pad_arrays_keep_invp(self.casual_trjs)
        
# 用于验证。模型训练好之后使用
def pad_arrays_keep_invp(src):
    """
    Pad arrays and return inverse permutation
    对位补齐，用于结果验证的时候
    Input:
    src (list[array[int32]])
    ---
    Output:
    src (seq_len, batch)
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    """
    idx = argsort(src) # [5, 7, 8, 9, 6, 1, 2, 0, 3, 4]
    src = list(np.array(src)[idx])
    lengths = list(map(len, src))  # [13, 13, 12, 12, 10, 5, 5, 4, 4, 3]
    lengths = torch.LongTensor(lengths)  
    ## 对位补齐
    src = pad_arrays(src) 
    invp = torch.LongTensor(invpermute(idx)) # [7, 5, 6, 8, 9, 0, 4, 1, 2, 3]  
    # 使其contiguous()在内存中连续
    return src.t().contiguous(), lengths.view(1, -1), invp

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
        
    
    ''' 根据时间段加载数据'''
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
            len(self.peak_trjs), len(self.work_trjs),len(self.casual_trjs)))
        return pad_arrays_keep_invp(self.peak_trjs),pad_arrays_keep_invp(self.work_trjs), \
    pad_arrays_keep_invp(self.casual_trjs)
    
    
    ''' 计算轨迹所属的时间段 '''
    def period(self,trj):
        cityname = constants.cityname
        scale = constants.scale
        time_size = constants.time_size
        if cityname == "beijing":
            lons_range, lats_range = constants.lons_range_bj,constants.lats_range_bj
        else:
            lons_range, lats_range = constants.lons_range_pt,constants.lats_range_pt
                
        '''获取在当前scale设定下，空间的划分'''
        maxx, maxy = (lons_range[1]-lons_range[0])//scale, (lats_range[1]-lats_range[0])//scale 
        space_size = maxx*maxy 
        time1 =  trj[0] // space_size
        time1 = (24* time1) //time_size 
        time2 = trj[-1] // space_size
        time2 = (24* time2) //time_size 
        if time1>=7 and time1<=10 and time2>=7 and time2<=10:
            return 1
        elif time1>=17 and time1<=20 and time2>=17 and time2<=20:
            return 1
        elif time1>=10 and time1<=17 and time2>=10 and time2<=17:
            return 2
        else:
            return 3
    
    
    ''' 获取一批固定数目轨迹， 返回 '''
    def getbatch(self, batch):
        """
        Output:
        src (seq_len, batch)
        lengths (1, batch)
        invp (batch,): inverse permutation, src.t()[invp] gets original order
        """
        if self.start >= self.size:
            return None, None, None
        src = self.srcdata[self.start:self.start+batch]
        ## update `start` for next batch
        self.start += self.batch
        self.start = 0
        return pad_arrays_keep_invp(src)
    
  
#test
if __name__ == "__main__":
    import constants
    from t2vec import setArgs
    from generate_groups import create_trj
    import os
    args = setArgs()
    trainsrc = os.path.join(args.data, "val.src")
    valData = DataLoader(trainsrc, args.batch)
    print("Reading training data...")
    valData.load(5000)
    batch = constants.val_batch
    epoch = len(valData.all_trjs)//batch
    co_pairs_set = [] # 迭代算法获得的所有co_pairs_set
    all_co_pairs = [] # 真正所有co_pairs_set
    all_groups_set = []
    trj2groupIDs_set = []
    for ii in tqdm(range(epoch)):
        valData.trjs = valData.all_trjs[ii*batch:(ii+1)*batch]
        co_pairs = valData.get_groups(constants.min_lifetime, constants.min_group_trj_nums)
        all_groups = valData.all_groups
        co_pairs_set.append(co_pairs)
        all_co_pairs.append(valData.get_pairs(constants.min_lifetime))
        all_groups_set.append(all_groups)
        trj2groupIDs_set.append(valData.trj2groupIDs)
    print(len(all_co_pairs))
    # 从每组选部分生成一组去编码
    # a, b = trainData.getbatch_one(each_nums=5)
    # a,p,n = trainData.getbatch_discriminative_inner()
    # a1,p1,n1 = trainData.getbatch_discriminative_common()
    

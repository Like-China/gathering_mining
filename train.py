import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models import EncoderDecoder
from data_utils import DataLoader
import time, os, shutil, logging
from tqdm import tqdm
import constants
from funcy import merge
from data_utils import pad_arrays_pair,pad_arrays_keep_invp
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
import numpy as np
from t2vec import setArgs
from evaluate import recall_after_combine, commom_vals
    
def init_parameters(model):
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)

def savecheckpoint(state, is_best, args):
    torch.save(state, args.checkpoint)
    if is_best:
        shutil.copyfile(args.checkpoint, os.path.join(args.data, 'best_model.pt'))

def loadTrainDataAndValidateDate(args):
    
    trainsrc = os.path.join(args.data, "train.src")
    trainData = DataLoader(trainsrc, args.batch)
    # print("Reading training data...")
    trainData.load(args.max_num_line)
    # print("Loaded train data size {}".format(trainData.size))
    valsrc  = os.path.join(args.data, "val.src")
    if os.path.isfile(valsrc):
        valData = DataLoader(valsrc, args.batch)
        # print("Reading validation data...")
        valData.load(args.read_val_nums)
        assert valData.size > 0, "Validation data size must be greater than 0"
        # print("Loaded validation data size {}".format(valData.size))
    else:
        valData = []
        # print("No validation data found, training without validating...")
    return trainData, valData


def disLoss(a, p, n, m0, triplet_loss, args):
    """
    a (named tuple): anchor data
    p (named tuple): positive data
    n (named tuple): negative data
    """
    ''' 
    计算相似性损失，即三角损失
    通过a,p,n三组轨迹，经过前向encoder,接着通过encoder_hn2decoder_h0，取最后一层向量作为每组每个轨迹的代表
'''
    # a_src (seq_len, 128)
    if(len(a.src)==0 or len(p.src)==0 or len(n.src) == 0):
        return 0,0
    a_src, a_lengths, a_invp = a.src, a.lengths, a.invp
    p_src, p_lengths, p_invp = p.src, p.lengths, p.invp
    n_src, n_lengths, n_invp = n.src, n.lengths, n.invp
    if args.cuda and torch.cuda.is_available():
        a_src, a_lengths, a_invp = a_src.cuda(), a_lengths.cuda(), a_invp.cuda()
        p_src, p_lengths, p_invp = p_src.cuda(), p_lengths.cuda(), p_invp.cuda()
        n_src, n_lengths, n_invp = n_src.cuda(), n_lengths.cuda(), n_invp.cuda()
    ## (num_layers * num_directions, batch, hidden_size)  (2*3, 128, 256/2)
    a_h, _ = m0.encoder(a_src, a_lengths)
    p_h, _ = m0.encoder(p_src, p_lengths)
    n_h, _ = m0.encoder(n_src, n_lengths)
    ## (num_layers, batch, hidden_size * num_directions) (3,128,256)
    a_h = m0.encoder_hn2decoder_h0(a_h)
    p_h = m0.encoder_hn2decoder_h0(p_h)
    n_h = m0.encoder_hn2decoder_h0(n_h)
    ## take the last layer as representations (batch, hidden_size * num_directions) (128,256)
    a_h, p_h, n_h = a_h[-1], p_h[-1], n_h[-1]
    ## 计算编码后 ap间的向量距离 是否比an间的向量距离更近
    # dis_ap = torch.mean(torch.abs(a_h[a_invp]-p_h[p_invp]),dim=1)
    # dis_an = torch.mean(torch.abs(a_h[a_invp]-n_h[n_invp]),dim=1)
    # diff = torch.mean(dis_ap-dis_an)
    return triplet_loss(a_h[a_invp], p_h[p_invp], n_h[n_invp])  # (128,256)

'''
    输入一组轨迹 a [src, lengths, invp]，从 每组随机抽取5个组成
    1. 前向计算得到代表向量
    2. 对代表向量进行聚类
    3. 选取a.p.n用于加快模型收敛
'''
def t2vec_cluster(trainData, m0, args):
    m0.eval()
    each_num = constants.each_num
    a_ids, a = trainData.selected_trj_ids,trainData.selected_trjs_pap
    if (len(a.src) == 0): return
    a_src, a_lengths, a_invp = a.src, a.lengths, a.invp
    if args.cuda and torch.cuda.is_available():
        a_src, a_lengths, a_invp = a_src.cuda(), a_lengths.cuda(), a_invp.cuda()
    a_h = []
    with torch.no_grad():
        a_h, _ = m0.encoder(a_src, a_lengths)
        a_h = m0.encoder_hn2decoder_h0(a_h)
        a_h = a_h[-1]
        a_h = a_h[a_invp]
    a_h = a_h.tolist()
    # # 开始聚类
    if args.c_method == 1:
        c = KMeans(n_clusters=constants.n).fit(a_h)
    elif    args.c_method == 2:
        c = DBSCAN(eps=constants.eps, min_samples=constants.mt).fit(a_h)
    else:
        c = AgglomerativeClustering(n_clusters=constants.n).fit(a_h)
    labels = c.labels_
    id2group = trainData.trj2groupIDs
    # 选取a.p.n的轨迹编号,记录轨迹编号
    apns = []
    for start in range(0,len(labels)-each_num,each_num):
        apn = []
        for ii in range(start,start+each_num):
            aa_id = a_ids[ii]
            aa_label = labels[ii]
            aa_group = id2group[aa_id]
            for jj in range(ii+1,start+each_num):
                bb_id = a_ids[jj]
                bb_label = labels[jj]
                # print(aa_label,bb_label)
                if (aa_label != bb_label):
                    nns_ids = np.array(a_ids)[labels==aa_id]
                    for nn_id in nns_ids:
                        nn_group = id2group[nn_id]
                        if (nn_group != aa_group):
                            apn.append([aa_id,bb_id,nn_id])
        if len(apn)>0:
            apns.append(apn)
        if len(apns) !=0 and len(merge(*apns))>args.max_apn_num:
            break
    
    m0.train()
    if (len(apns) == 0):
        print('no apn')
        return [],[],[]
    apns = merge(*apns)
    if len(apns) !=0 and len(merge(*apns))>args.max_apn_num:
        apns = apns[0:args.max_apn_num]
    # 进一步生成a p n轨迹对
    a_src,p_src,n_src = [],[],[]
    all_trjs = trainData.all_trjs
    for (a_id,p_id,n_id) in apns:
        a_src.append(all_trjs[a_id])
        p_src.append(all_trjs[p_id])
        n_src.append(all_trjs[n_id])
    a = pad_arrays_pair(a_src)
    p = pad_arrays_pair(p_src)
    n = pad_arrays_pair(n_src)
    return a,p,n

'''
将训练集样本全部转为向量
'''
def t2vec(args,m0,trainData):
    if torch.cuda.is_available():
        m0.cuda()
    m0.eval()
    vecs = []
    
    src, lengths, invp = pad_arrays_keep_invp(trainData.trjs)
    if torch.cuda.is_available():
        src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
    h, _ = m0.encoder(src, lengths) # 【层数*双向2，该组轨迹个数，隐藏层数】【6，10，128】
    h = m0.encoder_hn2decoder_h0(h)
    h = h.transpose(0, 1).contiguous()
    vecs.append(h[invp].cpu().data)
    
    vecs = torch.cat(vecs) # [10,3,256]
    vecs = vecs.transpose(0, 1).contiguous()  ## [3,10,256]
    # print("Encoding {} trjs...".format(len(vecs[m0.num_layers-1])))
    m0.train()
    return vecs[m0.num_layers-1].tolist()



def t2vec_cluster1(trainData, m0, args):
    if torch.cuda.is_available():
        m0.cuda()
    m0.eval()
    a_h = t2vec(args,m0,trainData)
    # # 开始聚类
    t1 = time.time()
    if args.c_method == 1:
        c = KMeans(n_clusters=constants.n).fit(a_h)
    elif    args.c_method == 2:
        c = DBSCAN(eps=constants.eps, min_samples=constants.mt).fit(a_h)
    else:
        c = AgglomerativeClustering(n_clusters=constants.n).fit(a_h)
    # print("cluster  "+str(time.time()-t1))
    labels = c.labels_
    id2group = trainData.trj2groupIDs
    G = trainData.all_groups
    # 选取a.p.n的轨迹编号,记录轨迹编号
    t1 = time.time()
    apns = []
    start = 0 #np.random.randint(0,len(G)//4)
    for ii in range(start,len(G)): # 对于每个组
        g = G[ii]
        apn = []
        for ii in range(len(g)):
            a_id = g[ii]
            a_label = labels[a_id]
            for jj in range(ii+1,len(g)):
                p_id = g[jj]
                p_label = labels[p_id]
                # 如果构成a.p对，再去查找n
                if (a_label != p_label):
                    # 聚类与a相同的轨迹编号
                    nns_ids = np.array(range(len(labels)))[labels==a_id]
                    for n_id in nns_ids:
                        nn_group = id2group[n_id]
                        if (nn_group != ii):
                            apn.append([a_id,p_id,n_id])
        
        if len(apn)>0:
            apns.append(apn)
        if len(apns) != 0 and len(merge(*apns))>args.max_apn_num:
            break
    # print("get a p n"+str(time.time()-t1))
    
    m0.train()
    if (len(apns) == 0):
        print('未有合适apn')
        return [],[],[]
    apns = merge(*apns)
    if len(apns) !=0 and len(merge(*apns))>args.max_apn_num:
        apns = apns[0:args.max_apn_num]
    a_src,p_src,n_src = [],[],[]
    all_trjs = trainData.all_trjs
    for (a_id,p_id,n_id) in apns:
        a_src.append(all_trjs[a_id])
        p_src.append(all_trjs[p_id])
        n_src.append(all_trjs[n_id])
    a = pad_arrays_pair(a_src)
    p = pad_arrays_pair(p_src)
    n = pad_arrays_pair(n_src)
    return a,p,n



'''
    验证获取genLoss, 这才是真正的验证集损失
'''
def validate(valData, m0, args, 
             co_pairs_set, all_groups_set, trj2groupIDs_set, all_co_pairs,cover_nums):
    
        
    ## switch to evaluation mode
    m0.eval()
    
    batch = constants.val_batch
    epoch = len(valData.all_trjs)//batch
    recalls = 0
    precs = 0
    # co_pair_match_ratios = 0
    
    n1 = 0 # 总的pairs数
    n2 = 0 # iter找到的
    n3 = 0 # model找到的pairs数
    for ii in range(epoch):
        t1 = time.time()
        valData.trjs = valData.all_trjs[ii*batch:(ii+1)*batch]
        valData.trj2groupIDs = trj2groupIDs_set[ii]
        valData.all_groups = all_groups_set[ii]
        # 获得分组
        # 获得co-pairs [i,j,common-token]
        # co_pairs = co_pairs_set[ii] # valData.get_groups(constants.min_lifetime, constants.min_group_trj_nums)
        a = pad_arrays_pair(valData.trjs)
        if (len(a.src) == 0): return
        a_src, a_lengths, a_invp = a.src, a.lengths, a.invp
        if args.cuda and torch.cuda.is_available():
            a_src, a_lengths, a_invp = a_src.cuda(), a_lengths.cuda(), a_invp.cuda()
        a_h = []
        with torch.no_grad():
            a_h, _ = m0.encoder(a_src, a_lengths)
            a_h = m0.encoder_hn2decoder_h0(a_h)
            a_h = a_h[-1]
            a_h = a_h[a_invp]
        a_h = a_h.tolist()
        
        if args.c_method == 1:
            c = KMeans(n_clusters=constants.n).fit(a_h)
        elif    args.c_method == 2:
            c = DBSCAN(eps=constants.eps, min_samples=constants.mt).fit(a_h)
        else:
            c = AgglomerativeClustering(n_clusters=constants.n).fit(a_h)
        labels = c.labels_
        print('model用时:')
        print(time.time()-t1)
        '''每一组聚类，co-movement pair的比例'''
        co_pair_num = 0
        al_pair_num = 0
        for i  in range(constants.n):
            # 取出每个聚类中的轨迹
            cluster = np.where(labels == i)[0].tolist()
            trjs = np.array(valData.trjs)[cluster].tolist()
            for mm in range(len(trjs)):
                for jj in range(mm+1,len(trjs)):
                    if (len(commom_vals(trjs[mm], trjs[jj]))>=constants.min_lifetime):
                        co_pair_num += 1
                    # elif valData.trj2groupIDs[cluster[mm]] == -1:
                    #     pass
                    # elif valData.trj2groupIDs[cluster[jj]] == -1:
                    #     pass
                    else:
                        al_pair_num += 1
        precs += co_pair_num/al_pair_num
        ''' 对于所有的co-movement pair ,落在同一个聚类的比例'''
        # pair_match_num = 0
        # for co_pair in co_pairs:
        #     if labels[co_pair[0]] == labels[co_pair[1]]:
        #         pair_match_num += 1
        # co_pair_match_ratios += pair_match_num/len(co_pairs)
        ''' 击中的co-pairs数目比较'''
        n1 += len(all_co_pairs[ii])
        n2 += cover_nums[ii]
        n3 += co_pair_num
        
        ''' 计算召回率'''
        # all_groups1, all_group_cvs1, t2g2 = group_by_cluster(labels,valData.all_trjs)
        recall = recall_after_combine(labels, valData.all_groups) # 仅聚类+重新分组后的recall
        recalls += recall
        
    print('总的co-pairs:{0}, iter找到:{1}, cluster找到:{2}'.format(n1//epoch, n2//epoch,n3//epoch))
    # print("验证集轨迹 mean pair matching ratio={0:.3f}".format(co_pair_match_ratios/epoch))
    print("验证集轨迹 mean co-movement prec={0:.3f}".format(precs/epoch))
    # print("验证集平均召回率：{0:.3f}".format(recalls/epoch))
    ## switch back to training mode
    m0.train()
    return recalls/epoch



def train(args):
    logging.basicConfig(filename=os.path.join(args.data, "training.log"), level=logging.INFO)
    trainData, valData = loadTrainDataAndValidateDate(args)
    # 训练集trjs为随机选取batch_size个，验证集为all_trjs
    trainData.getbatch_one(constants.each_num, False)
    # valData.getbatch_one(constants.each_num, True)
    
    # 提前计算所有val batch如每5000条验证集轨迹的co-pairs
    print('提前计算所有val batch条验证集轨迹的co-pairs')
    batch = constants.val_batch
    epoch = len(valData.all_trjs)//batch
    co_pairs_set = [] # 迭代算法获得的所有co_pairs_set
    all_co_pairs = [] # 真正所有co_pairs_set
    all_groups_set = []
    trj2groupIDs_set = []
    cover_nums = [] # 计算基于迭代的算法cover了多少co-pair
    for ii in range(epoch):
        valData.trjs = valData.all_trjs[ii*batch:(ii+1)*batch]
        co_pairs = valData.get_groups(constants.min_lifetime, constants.min_group_trj_nums)
        all_groups = valData.all_groups
        cover_nums.append(valData.cover_num)
        co_pairs_set.append(co_pairs)
        # all_co_pairs.append(valData.get_pairs(constants.min_group_trj_nums,constants.min_lifetime)) #
        all_co_pairs.append([]) 
        all_groups_set.append(all_groups)
        trj2groupIDs_set.append(valData.trj2groupIDs)
    
    cityname = constants.cityname
    scale = constants.scale
    time_size = constants.time_size
    if cityname == "beijing":
        lons_range, lats_range = constants.lons_range_bj,constants.lats_range_bj
    else:
        lons_range, lats_range = constants.lons_range_pt,constants.lats_range_pt
        
    '''获取在当前scale设定下，空间的划分'''
    minx, miny = 0,0
    maxx, maxy = (lons_range[1]-lons_range[0])//scale, (lats_range[1]-lats_range[0])//scale
    vocal_size = max(trainData.maxID, valData.maxID)+8
    # vocal_size = int(maxx*maxy*time_size+maxx*maxy)+8
    print("vocal_size= {}".format(vocal_size))
    print(trainData.minID)
    print(valData.minID)
    # 创建损失函数
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    # 输入到输出整个encoder-decoder 的 map
    m0 = EncoderDecoder(vocal_size,
                        args.embedding_size,
                        args.hidden_size,
                        args.num_layers, 
                        args.dropout, 
                        args.bidirectional)
    if args.cuda and torch.cuda.is_available():
        print("=> training with GPU")
        m0.cuda()
    else:
        print("=> training with CPU")

    m0_optimizer = torch.optim.Adam(m0.parameters(), 
                                    lr=args.learning_rate)

    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        logging.info("Restore training @ {}".format(time.ctime()))
        checkpoint = torch.load(args.checkpoint)
        args.start_iteration = checkpoint["iteration"]
        best_recall = checkpoint["best_recall"]
        best_train_loss = checkpoint["best_train_loss"]
        
        m0.load_state_dict(checkpoint["m0"])
        m0_optimizer.load_state_dict(checkpoint["m0_optimizer"])
        
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        logging.info("Start training @ {}".format(time.ctime()))
        best_recall = float('0')
        best_train_loss = float('inf')
        print("=> initializing the parameters...")
        init_parameters(m0)
        ## here: load pretrained wrod (cell) embedding

    print("开始训练："+str(time.ctime()))
    print("Iteration starts at {0} and will end at {1:.4f} \n".format(args.start_iteration, args.iter_num-1))
    
    
    # 用一个计数器 计数 测试集损失 未下降的次数，若超过一定次数，则直接停止训练
    invalid_count = 0
    for iteration in range(args.start_iteration+1, args.iter_num):
        try:
            # 梯度初始化为0
            m0_optimizer.zero_grad()
            disloss_inner, disloss_common = torch.tensor(0),torch.tensor(0)
            t_loss,t_loss1 = torch.tensor(0),torch.tensor(0)
            # 添加一组损失计算
            if  iteration % 21 == 0:
                trainData.getbatch_one(constants.each_num, False)
                
                
            a, p, n = [],[],[]#2vec_cluster(trainData, m0, args)
            # print(iteration)
            
            if len(a) != 0:
                t_loss = disLoss(a, p, n, m0, triplet_loss, args)
                if iteration % args.print_freq == 0:
                    print("选取apn数:{}".format(len(a.src[0])))
                
            a1, p1, n1 = t2vec_cluster1(trainData, m0, args)
            if len(a1) != 0:
                t_loss1 = disLoss(a1, p1, n1, m0, triplet_loss, args)
                if iteration % args.print_freq == 0:
                    print("选取apn1数:{}\n".format(len(a1.src[0])))
            else:
                trainData.getbatch_one(constants.each_num, False)
            #a, p, n = trainData.getbatch_discriminative_inner()
           # disloss_inner = disLoss(a, p, n, m0, triplet_loss, args)
            # 新增(2021/8/17) a.p.n  按照公共元素个数划分
          #  a, p, n= trainData.getbatch_discriminative_common()
           # print(a.src)
          #  disloss_common = disLoss(a, p, n, m0, triplet_loss, args)
            
            loss = t_loss1 #+ t_loss# + 0.2*(disloss_common + disloss_inner)
            if loss.item() == 0:
                continue
            ## 根据模型损失，计算梯度
            loss.backward()
            ## 限制梯度下降的阈值，防止梯度消失现象,更新全部参数一次
            clip_grad_norm_(m0.parameters(), args.max_grad_norm)
            m0_optimizer.step()
            ''' 更改：存储三元损失更小的训练模型 (2021/8/18)'''
            ## 定期输出训练状态
            if iteration % args.print_freq == 0:
                print ("\n\n当前时间:"+str(time.ctime()))
                print("Iteration: {0:}"\
                          "\nTrain Triplet Loss: {1:.3f}"\
                      "\nTrain Triplet Loss1: {2:.3f}"\
                          "\nDiscriminative Train Inner Loss: {3:.3f}"\
                      "\nDiscriminative Train Common Loss: {4:.3f}"\
            .format(iteration, t_loss, t_loss1, disloss_inner,disloss_common))
                print("best_train_loss= {0:.3f}".format(best_train_loss))
                
            ## 定期存储训练状态，通过验证集前向计算当前模型损失，若能获得更小损失，则保存最新的模型参数
            
            if iteration % args.save_freq == 0 and iteration > 0:
                # 如果训练集能够取得更好的模型，再进一步进行验证集验证
                ''' 更改：存储三元损失更小的训练模型 (2021/8/18)'''
                    
                recall = validate(valData, m0, args, co_pairs_set, 
                                  all_groups_set, trj2groupIDs_set, all_co_pairs,
                                  cover_nums)
                print("current recall:{}\nbest recall: {}".
                      format(recall,best_recall))
                ## 如果测试集或训练集损失 很多次都没有减少，则停止训练
                ## 若有减少，则存储为best_model
                if recall > best_recall  or (t_loss1.item() !=0 and  t_loss1.item()< best_train_loss):
                    if t_loss1.item() !=0 and  t_loss1.item()< best_train_loss:
                        best_train_loss = t_loss1.item()
                    best_recall = recall
                    logging.info("Best model with recall {} at iteration {} @ {}"\
                                 .format(best_recall, iteration, time.ctime()))
                    is_best = True
                    invalid_count = 0
                else:
                    is_best = False
                    invalid_count += 1
                    # if(invalid_count>=10):
                    #     break
                if is_best:
                    print("Saving the model at iteration {0} with best recall {1:.2f}".format(iteration, best_recall))
                    savecheckpoint({
                        "iteration": iteration,
                        "best_recall": best_recall,
                        "m0": m0.state_dict(),
                        "m0_optimizer": m0_optimizer.state_dict(),
                        "best_train_loss":best_train_loss
                    }, is_best, args)
            
        except KeyboardInterrupt:
            break



def test(args, m0, val_num, min_lifetime, min_group_trj_nums,n, is_period = False):
    trainData, valData = loadTrainDataAndValidateDate(args)
    # 提前计算所有val batch如每5000条验证集轨迹的co-pairs
    print('val_num = {0} min_lifetime={1} min_group_trj_nums={2} \
          cluster_n={3}'.format(val_num, min_lifetime, min_group_trj_nums,n))
    batch = val_num
    epoch = 1
    co_pairs_set = [] # 迭代算法获得的所有co_pairs_set
    all_co_pairs = [] # 真正所有co_pairs_set
    all_groups_set = []
    trj2groupIDs_set = []
    cover_nums = [] # 计算基于迭代的算法cover了多少co-pair
    
    # 验证不同时刻的 co-movement pair数量
    
    for ii in range(epoch):
        if is_period:
            valData = trainData
            valData.load_by_period()
            select_num = min(len(valData.peak_trjs),
                             len(valData.work_trjs),
                             len(valData.casual_trjs))
            # valData.trjs = np.array(valData.peak_trjs)[0:select_num].tolist() # valData.work_trjs[select_num] 
            # valData.trjs = np.array(valData.work_trjs)[0:select_num].tolist() 
            valData.trjs = np.array(valData.casual_trjs)[0:select_num].tolist() 
        else:
            valData.trjs = valData.all_trjs[ii*batch:(ii+1)*batch]
        # 
        # 吧valData.trjs指定为valData.peak_trjs,valData.work_trjs,valData.casual_trjs即可
        t1 = time.time()
        co_pairs = valData.get_groups(min_lifetime, min_group_trj_nums)
        print('iter算法用时:'+str(time.time()-t1))
        all_groups = valData.all_groups
        cover_nums.append(valData.cover_num)
        co_pairs_set.append(co_pairs)
        all_co_pairs.append([])
        # all_co_pairs.append(valData.get_pairs(min_group_trj_nums,min_lifetime))
        all_groups_set.append(all_groups)
        trj2groupIDs_set.append(valData.trj2groupIDs)
    
    ''' 开始计算覆盖率'''    
    n1 = 0 # 总的pairs数
    n2 = 0 # iter找到的
    n3 = 0 # model找到的pairs数
    for ii in range(epoch):
        t1 = time.time()
        # valData.trjs = valData.all_trjs[ii*batch:(ii+1)*batch]
        valData.trj2groupIDs = trj2groupIDs_set[ii]
        valData.all_groups = all_groups_set[ii]
        a = pad_arrays_pair(valData.trjs)
        if (len(a.src) == 0): return
        a_src, a_lengths, a_invp = a.src, a.lengths, a.invp
        if args.cuda and torch.cuda.is_available():
            a_src, a_lengths, a_invp = a_src.cuda(), a_lengths.cuda(), a_invp.cuda()
        a_h = []
        with torch.no_grad():
            a_h, _ = m0.encoder(a_src, a_lengths)
            a_h = m0.encoder_hn2decoder_h0(a_h)
            a_h = a_h[-1]
            a_h = a_h[a_invp]
        a_h = a_h.tolist()
        
        if args.c_method == 1:
            c = KMeans(n_clusters=n).fit(a_h)
        labels = c.labels_
        print('model用时:'+str(time.time()-t1))
        '''每一组聚类，co-movement pair的比例'''
        co_pair_num = 0
        al_pair_num = 0
        for i  in range(n):
            # 取出每个聚类中的轨迹
            cluster = np.where(labels == i)[0].tolist()
            trjs = np.array(valData.trjs)[cluster].tolist()
            for mm in range(len(trjs)):
                for jj in range(mm+1,len(trjs)):
                    if (len(commom_vals(trjs[mm], trjs[jj]))>=min_lifetime):
                        co_pair_num += 1
                    else:
                        al_pair_num += 1
        ''' 击中的co-pairs数目比较'''
        n1 += len(all_co_pairs[ii])
        n2 += cover_nums[ii]
        n3 += co_pair_num
    print('总的co-pairs:{0}, iter找到:{1}, cluster找到:{2}\n'.format(n1//epoch, n2//epoch,n3//epoch))
    
''' 测试生成apn用时'''
if __name__ == "__main__":
    args = setArgs()
    trainData, valData = loadTrainDataAndValidateDate(args)
    '''获取在当前scale设定下，空间的划分'''
    cityname = constants.cityname
    scale = constants.scale
    time_size = constants.time_size
    if cityname == "beijing":
        lons_range, lats_range = constants.lons_range_bj,constants.lats_range_bj
    else:
        lons_range, lats_range = constants.lons_range_pt,constants.lats_range_pt
    minx, miny = 0,0
    maxx, maxy = (lons_range[1]-lons_range[0])//scale, (lats_range[1]-lats_range[0])//scale
    if cityname == "beijing":
        vocal_size = int(maxx*maxy*time_size+maxx*maxy)+8
    else:
        vocal_size = max(trainData.maxID, valData.maxID)+8
    print("vocal_size= {}".format(vocal_size))
    m0 = EncoderDecoder(7439, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        if torch.cuda.is_available():
            m0.cuda()
    
    
    val_num = 10000
    min_lifetime = 5
    min_group_trj_nums = 10
    n = 10
    # print("默认设置")
    test(args, m0, val_num, min_lifetime, min_group_trj_nums,n)
    
    # print('变化min_lifetime')
    # for ms in [5,10,15,20]:
    #     test(args, m0, val_num, ms, min_group_trj_nums,n )
    # print('变化min_group_trj_nums')
    # for mn in [10,20,30,40]:
    #     test(args, m0, val_num, min_lifetime, mn,n )
    # print('变化cluster num:')
    # for nn in [10,20,30,40,50]:
    #     test(args, m0, val_num, min_lifetime, min_group_trj_nums,nn)
    # print('变化val_num:')
    # for num in [5000,10000,15000,20000]:
    #     test(args, m0, num, min_lifetime, min_group_trj_nums,n )
    # print('不同时间段')
    # test(args, m0, val_num, min_lifetime, min_group_trj_nums,n,True)
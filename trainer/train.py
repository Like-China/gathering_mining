import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from trainer.models import EncoderDecoder
from loader.data_utils import pad_arrays_pair, pad_arrays_keep_invp
from loader.data_loader import DataLoader
import time
import os
import shutil
import logging
import settings as constants
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import numpy as np
from index.scan import  get_communities
from baselines.GS_ACMC import ACMC
from tqdm import tqdm


def init_parameters(model):
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)


def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint)
    if is_best:
        shutil.copyfile(args.checkpoint, os.path.join(args.data, 'best_model.pt'))


def load_data(args):
    train_loader = DataLoader(os.path.join(args.data, "train"), args.batch)
    train_loader.load(args.max_train_num)
    validate_loader = DataLoader(os.path.join(args.data, "val"), args.batch)
    validate_loader.load(args.max_val_nums)
    assert validate_loader.size > 0, "Validation data size must be greater than 0"
    print("Loaded train data size {}".format(train_loader.size))
    print("Loaded validation data size {}".format(validate_loader.size))
    return train_loader, validate_loader


def get_triplet_loss(a, p, n, m0, triplet_loss, args):
    """
    Calculate the similarity loss, namely triplet loss
    Through a,p,n groups trajectory, forward encoder, then through encoder_hn2decoder_h0，get vector of the last layer represent each trajectory of each group

    a (named tuple): anchor data
    p (named tuple): positive data
    n (named tuple): negative data
    """
    # a_src (seq_len, 128)
    if len(a.src) == 0:
        return None
    a_src, a_lengths, a_invp = a.src, a.lengths, a.invp
    p_src, p_lengths, p_invp = p.src, p.lengths, p.invp
    n_src, n_lengths, n_invp = n.src, n.lengths, n.invp
    if args.cuda and torch.cuda.is_available():
        a_src, a_lengths, a_invp = a_src.cuda(), a_lengths.cuda(), a_invp.cuda()
        p_src, p_lengths, p_invp = p_src.cuda(), p_lengths.cuda(), p_invp.cuda()
        n_src, n_lengths, n_invp = n_src.cuda(), n_lengths.cuda(), n_invp.cuda()
    # (num_layers * num_directions, batch, hidden_size)  (2*3, 128, 256/2)
    a_h, _ = m0.encoder(a_src, a_lengths)
    p_h, _ = m0.encoder(p_src, p_lengths)
    n_h, _ = m0.encoder(n_src, n_lengths)
    # (num_layers, batch, hidden_size * num_directions) (3,128,256)
    a_h = m0.encoder_hn2decoder_h0(a_h)
    p_h = m0.encoder_hn2decoder_h0(p_h)
    n_h = m0.encoder_hn2decoder_h0(n_h)
    # take the last layer as representations (batch, hidden_size * num_directions) (128,256)
    a_h, p_h, n_h = a_h[-1], p_h[-1], n_h[-1]
    # Calculate whether the vector distance between ap after coding is closer than that between an
    dis_ap = torch.mean(torch.abs(a_h[a_invp]-p_h[p_invp]), dim=1)
    dis_an = torch.mean(torch.abs(a_h[a_invp]-n_h[n_invp]), dim=1)
    diff = torch.mean(dis_ap-dis_an)
    # print("The difference between vectors: ", diff.item())
    return diff, triplet_loss(a_h[a_invp], p_h[p_invp], n_h[n_invp])  # (128,256)


def t2vec(m0, token_sequences):
    """
    Convert all training set samples into vectors

    :param m0: training model
    :param token_sequences: trajectory token sequence
    :return: The vector representation of the last layer
    """
    m0.eval()
    res = []
    src, lengths, invp = pad_arrays_keep_invp(token_sequences)
    if torch.cuda.is_available():
        src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
    h, _ = m0.encoder(src, lengths)  # [Layers*2，Number of trajectories in this group，Number of hidden layers][6，10，128]
    h = m0.encoder_hn2decoder_h0(h)
    h = h.transpose(0, 1).contiguous()
    res.append(h[invp].cpu().data)

    res = torch.cat(res)  # [10,3,256]
    res = res.transpose(0, 1).contiguous()  # [3,10,256]
    m0.train()
    return res[m0.num_layers - 1].tolist()


# 快速计算欧式距离
def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


# 输入一组轨迹，编码成向量被输出聚类标签
def vec_cluster(trjs, m0, args):
    a_h = t2vec(m0, trjs)
    if args.c_method == 1:
        c = KMeans(n_clusters=constants.n).fit(a_h)
    elif args.c_method == 2:
        c = DBSCAN(eps=constants.ep, min_samples=constants.min_group_trj_nums).fit(a_h)
    elif args.c_method == 3:
        c = AgglomerativeClustering(n_clusters=constants.n).fit(a_h)
    else:
        pairs = []
        # 查找一组CLS>min_length的轨迹对的距离作为下面的参考值
        anchor_value = []
        count = 100
        a_h = torch.tensor(a_h)
        dists = euclidean_dist(a_h, a_h)
        for ii in tqdm(desc='get anchor', len(trjs)):
            for jj in range(ii+1, len(trjs)):
                trj1, trj2 = trjs[ii], trjs[jj]
                cvals1= list(set(trj1).intersection(set(trj2)))
                if len(cvals1)>constants.min_lifetime:
                    count +=1
                    anchor_value.append(dists[ii][jj])
        anchor_value = np.mean(anchor_value)
        print("anchor={}".format(anchor_value))


        for ii in tqdm(range(len(dists))):
            for jj in range(ii + 1, len(dists)):
                dist = dists[ii][jj]
                if dist <= anchor_value:
                    pairs.append([ii, jj])
        print("找到Pair数: {}".format(len(pairs)))
        communities, hubs, outliers = get_communities(pairs, constants.min_group_trj_nums, constants.ep)
        labels = [-1] * len(a_h)
        for ii in range(len(communities)):
            for trj_id in communities[ii]:
                labels[trj_id] = ii
        print(
            '归组人数:{0}\t未归组人数：{1}\t组数：{2}'.format(sum(np.array(labels) != -1), sum(np.array(labels) == -1),
                                                 len(communities)))
    return labels if args.c_method == 0 else c.labels_


def get_apn(trjs, m0, args):
    """
    Enter a set of trajectories a[src, lengths, invp], 5 components were randomly selected from each group
        1. The forward calculation get the representative vector
        2. Cluster representative vectors
        3. select a.p.n for accelerate the model convergence
            3.1 a-p They belong to the same group but are clustered into different classes
            3.2 a-n Not in the same group but grouped into the same group
    """
    # m0.eval()
    cluster_start_time = time.time()
    labels = vec_cluster(trjs, m0, args)
    print("Clustering time :  "+str(time.time()-cluster_start_time))
    # apn_start_time = time.time()
    apns = []
    for i in range(constants.n):
        cluster = np.where(labels == i)[0].tolist()
        non_cluster = np.where(labels != i)[0].tolist()
        trj_cluster = np.array(trjs)[cluster].tolist()
        non_trj_cluster = np.array(trjs)[non_cluster].tolist()
        for mm in range(len(trj_cluster)):
            for jj in range(mm + 1, len(trj_cluster)):
                common_num1 = list(set(trj_cluster[mm]).intersection(set(trj_cluster[jj])))
                if len(common_num1) < constants.min_lifetime:
                    # an = [cluster[mm], cluster[jj]]
                    for nn in range(len(non_trj_cluster)):
                        common_num2 = list(set(trj_cluster[mm]).intersection(set(non_trj_cluster[nn])))
                        if len(common_num2) >= constants.min_lifetime:
                            apns.append([cluster[mm], non_cluster[nn], cluster[jj]])
            if len(apns) > args.max_apn_num: break
        if len(apns) > args.max_apn_num: break

    # print("get a p n"+str(time.time()-apn_start_time))
    # m0.train()
    if len(apns) == 0:
        print('no valid apn')
        return [], [], []
    a_src, p_src, n_src = [], [], []
    for a_id, p_id, n_id in apns[0:args.max_apn_num]:
        a_src.append(trjs[a_id])
        p_src.append(trjs[p_id])
        n_src.append(trjs[n_id])
    a = pad_arrays_pair(a_src)
    p = pad_arrays_pair(p_src)
    n = pad_arrays_pair(n_src)
    return a, p, n


def validate(validate_loader, m0,  args, real_labels):
    """
    Validation capture genLoss, which is the true validation set loss
    """
    m0.eval()
    recalls = []
    for epoch in range(validate_loader.size // constants.val_batch):
        # 当前轮次的真实标签
        real_label = real_labels[epoch]
        if real_label is None: continue
        # t1 = time.time()
        trjs = validate_loader.all_trjs[epoch * constants.val_batch:(epoch + 1) * constants.val_batch]
        # 得到聚类标签
        predict_label = vec_cluster(trjs, m0, args)
        # 未满人数限制的全部记录为-1
        predict_label = np.array(predict_label)

        valid_cluster_count = 0
        for item in set(predict_label):
            class_index = np.where(predict_label==item)
            if len(class_index)<constants.min_group_trj_nums:
                predict_label[class_index] = -1
            else:
                valid_cluster_count
        print("预测划分为{}类, 有效类为{}类".format(len(set(predict_label)),valid_cluster_count))


        # 计算召回率
        TP, FP, TN, FN = 0, 0, 0, 0
        for ii in range(len(real_label)):
            for jj in range(ii+1, len(real_label)):
                if real_label[ii] == real_label[jj]:
                    if predict_label[ii] == predict_label[jj]:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if predict_label[ii] == predict_label[jj]:
                        FP += 1
                    else:
                        TN += 1
        print(TP, FN, FP, TN)
        recalls.append(TP/(TP+FN))
    print("Average recall rate of validation sets： {0:.3f}".format(np.mean(recalls)))
    m0.train()
    return np.mean(recalls) * 100


def train(args):
    logging.basicConfig(filename=os.path.join(args.data, "training.log"), level=logging.INFO)
    train_loader, validate_loader = load_data(args)
    vocal_size = max(train_loader.maxID, validate_loader.maxID)+1
    print("vocal_size= {} min_id = {} max_id = {}".format(vocal_size,
                min(train_loader.minID, validate_loader.minID), max(train_loader.maxID, validate_loader.maxID)))
    # 对于每一批，先使用ACMC获得组划分ground-truth, 只用计算一次(LCS+SCAN)
    real_labels = []
    for ii in range(validate_loader.size // constants.val_batch):
        validate_batch = validate_loader.all_trjs[ii * constants.val_batch:(ii + 1) * constants.val_batch]
        real_label = ACMC().get_groups(validate_batch)
        real_labels.append(real_label)
    triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2)
    m0 = EncoderDecoder(args, vocal_size)
    if args.cuda and torch.cuda.is_available():
        print("=> training with GPU")
        m0.cuda()
    m0_optimizer = torch.optim.Adam(m0.parameters(), lr=args.learning_rate)
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

    print("Starting training ：" + str(time.ctime()))
    print("Iteration starts at {0} and will end at {1:.4f} \n".format(args.start_iteration, args.iter_num - 1))
    invalid_count = 0
    for iteration in range(args.start_iteration + 1, args.iter_num):
        try:
            m0_optimizer.zero_grad()
            inner_loss, common_loss = torch.tensor(0), torch.tensor(0)
            t_loss, t_loss1 = torch.tensor(0), torch.tensor(0)

            train_batch = train_loader.get_batch()

            a, p, n = get_apn(train_batch, m0, args)
            if len(a) != 0:
                diff, t_loss = get_triplet_loss(a, p, n, m0, triplet_loss, args)
                if iteration % args.print_freq == 0:
                    print("The number of apn:{0}  difference ： {1}".format(len(a.src[0]), diff))

            # a1, p1, n1 = train_loader.get_apn_cross()
            # if len(a) != 0:
            #     diff, t_loss1 = get_triplet_loss(a1, p1, n1, m0, triplet_loss, args)
            #     if iteration % args.print_freq == 0:
            #         print("The number of apn:{0}  difference ： {1}".format(len(a1.src[0]), diff))

            # a2, p2, n2 = train_loader.get_inner_apn()
            # if len(a2) != 0:
            #     diff, inner_loss = get_triplet_loss(a2, p2, n2, m0, triplet_loss, args)
            #     if iteration % args.print_freq == 0:
            #         print("The number of apn:{0}  difference ： {1}".format(len(a2.src[0]), diff))

            # a3, p3, n3 = train_loader.get_common_apn()
            # if len(a3) != 0:
            #     diff, common_loss = get_triplet_loss(a3, p3, n3, m0, triplet_loss, args)
            #     if iteration % args.print_freq == 0:
            #         print("The number of apn:{0}  difference ： {1}".format(len(a3.src[0]), diff))
            loss = 0.5*(t_loss + t_loss1) + 0.5*(inner_loss + common_loss)
            if loss.item() == 0:
                continue
            clip_grad_norm_(m0.parameters(), args.max_grad_norm)
            m0_optimizer.step()
            if iteration % args.print_freq == args.print_freq - 1:
                print("\n\n Current time :" + str(time.ctime()))
                print("Iteration: {}".format(iteration))
                print("Train Triplet Loss: {0:.6f}".format(t_loss))
                print("Train Triplet Loss1: {0:.6f}".format(t_loss1))
                print("Discriminative Train Inner Loss: {0:.6f}".format(inner_loss))
                print("Discriminative Train Common Loss: {0:.6f}".format(common_loss))
                print("Total Train Loss: {0:.6f}".format(loss))
                print("best_train_loss= {0:.6f}".format(best_train_loss))
            if iteration % args.save_freq == args.save_freq - 1:
                is_best = False
                if loss.item() != 0 and loss.item() < best_train_loss:
                    recall = validate(validate_loader, m0, args, real_labels)
                    print("current recall:{}\n best recall: {}".format(recall, best_recall))
                    if recall > best_recall:
                        best_train_loss = loss.item()
                        best_recall = recall
                        logging.info("Best recall {} at iteration {} @ {}".format(best_recall, iteration, time.ctime()))
                        is_best = True
                        invalid_count = 0
                else:
                    is_best = False
                    invalid_count += 1
                    if invalid_count >= args.max_invalid_num:
                        break
                if is_best:
                    print("Saving the model at iteration {0} with best recall {1:.2f}".format(iteration, best_recall))
                    save_checkpoint({
                        "iteration": iteration,
                        "best_recall": best_recall,
                        "m0": m0.state_dict(),
                        "m0_optimizer": m0_optimizer.state_dict(),
                        "best_train_loss": best_train_loss
                    }, is_best, args)

        except KeyboardInterrupt:
            break

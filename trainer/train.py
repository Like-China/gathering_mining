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
from data.porto.porto1500350.evaluator.evaluate import get_recall
from baselines.GS_ACMC import ACMC


def common_values(m, n):
    """
    Find the common sequence of two ordered sequences
    Some pre-checking strategy base of min_lifetime 
    """
    len_m = len(m)
    len_n = len(n)
    # Two Pointers to two arrays
    i, j = 0, 0
    # Recording common values
    common_value = []
    while i < len_m and j < len_n:
        if m[i] == n[j]:
            common_value.append(m[i])
            i += 1
            j += 1
        elif m[i] < n[j]:
            i += 1
        else:
            j += 1
    return common_value


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
    a_h = t2vec(m0, trjs)
    # cluster_start_time = time.time()
    if args.c_method == 1:
        c = KMeans(n_clusters=constants.n).fit(a_h)
    elif args.c_method == 2:
        c = DBSCAN(eps=constants.eps, min_samples=constants.mt).fit(a_h)
    else:
        c = AgglomerativeClustering(n_clusters=constants.n).fit(a_h)
    # print("Clustering time :  "+str(time.time()-cluster_start_time))
    labels = c.labels_
    # apn_start_time = time.time()
    apns = []
    for i in range(constants.n):
        cluster = np.where(labels == i)[0].tolist()
        non_cluster = np.where(labels != i)[0].tolist()
        trj_cluster = np.array(trjs)[cluster].tolist()
        non_trj_cluster = np.array(trjs)[non_cluster].tolist()
        for mm in range(len(trj_cluster)):
            for jj in range(mm + 1, len(trj_cluster)):
                if len(common_values(trj_cluster[mm], trj_cluster[jj])) < constants.min_lifetime:
                    # an = [cluster[mm], cluster[jj]]
                    for nn in range(len(non_trj_cluster)):
                        if len(common_values(trj_cluster[mm], non_trj_cluster[nn])) > constants.min_lifetime:
                            apns.append([cluster[mm], non_cluster[nn], cluster[jj]])
            if len(apns) > args.max_apn_num:
                break
        if len(apns) > args.max_apn_num:
            break
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


def get_apn1(trjs, m0, args):
    """
    Enter a set of trajectories a[src, lengths, invp], 5 components were randomly selected from each group
        1. The forward calculation get the representative vector
        2. Cluster representative vectors
        3. select a.p.n for accelerate the model convergence
            3.1 a-p They belong to the same group but are clustered into different classes
            3.2 a-n Not in the same group but grouped into the same group
    """
    # m0.eval()

    a_h = t2vec(m0, trjs)
    # cluster_start_time = time.time()
    if args.c_method == 1:
        c = KMeans(n_clusters=constants.n).fit(a_h)
    elif args.c_method == 2:
        c = DBSCAN(eps=constants.eps, min_samples=constants.mt).fit(a_h)
    else:
        c = AgglomerativeClustering(n_clusters=constants.n).fit(a_h)
    # print("clustering time:  "+str(time.time()-cluster_start_time))
    labels = c.labels_

    group_start_time = time.time()
    greedy = ACMC()
    pairs, groups, trj_map_group = greedy.get_groups(trjs)
    print("ACMC group time: ", time.time() - group_start_time)
    # apn_start_time = time.time()
    apns = []
    start = 0
    for group_id in range(start, len(groups)):
        g = groups[group_id]
        apn = []
        for ii in range(len(g)):
            a_id = g[ii]
            a_label = labels[a_id]
            for jj in range(ii + 1, len(g)):
                p_id = g[jj]
                p_label = labels[p_id]
                if a_label != p_label:
                    nns_ids = np.array(range(len(labels)))[labels == a_id]
                    for n_id in nns_ids:
                        nn_group = trj_map_group[n_id]
                        if nn_group != group_id:
                            apn.append([a_id, p_id, n_id])
        if len(apn) > 0:
            apns.append(apn)
        if len(apns) != 0 and len(sum(apns, [])) > args.max_apn_num:
            break
    # print("get a p n"+str(time.time()-apn_start_time))
    m0.train()

    if len(apns) == 0:
        print('no valid apn')
        return [], [], []
    apns = sum(apns, [])
    a_src, p_src, n_src = [], [], []
    for a_id, p_id, n_id in apns[0:args.max_apn_num]:
        a_src.append(trjs[a_id])
        p_src.append(trjs[p_id])
        n_src.append(trjs[n_id])
    a = pad_arrays_pair(a_src)
    p = pad_arrays_pair(p_src)
    n = pad_arrays_pair(n_src)
    return a, p, n


def validate(validate_loader, m0, args, all_groups):
    """
    Validation capture genLoss, which is the true validation set loss
    """
    m0.eval()
    epoch = validate_loader.size // constants.val_batch
    recalls = []
    precisions = []

    for ii in range(epoch):
        if len(all_groups[ii]) == 0:
            continue
        grouped_ids = sum(all_groups[ii], [])
        # t1 = time.time()
        trjs = validate_loader.all_trjs[ii * constants.val_batch:(ii + 1) * constants.val_batch]

        a_h = t2vec(m0, trjs)
        if args.c_method == 1:
            c = KMeans(n_clusters=constants.n).fit(a_h)
        elif args.c_method == 2:
            c = DBSCAN(eps=constants.eps, min_samples=constants.mt).fit(a_h)
        else:
            c = AgglomerativeClustering(n_clusters=constants.n).fit(a_h)
        labels = c.labels_
        # print('model time :', time.time()-t1)

        co_pair_num = 0
        al_pair_num = 0
        for i in range(constants.n):
            cluster = np.where(labels == i)[0].tolist()
            trj_cluster = np.array(trjs)[cluster].tolist()
            for mm in range(len(trj_cluster)):
                if cluster[mm] not in grouped_ids:
                    continue
                for jj in range(mm + 1, len(trj_cluster)):
                    if cluster[jj] not in grouped_ids:
                        continue
                    if len(common_values(trj_cluster[mm], trj_cluster[jj])) >= constants.min_lifetime:
                        co_pair_num += 1
                    al_pair_num += 1
        precisions.append(co_pair_num / max(1, al_pair_num))

        recalls.append(get_recall(labels, all_groups[ii]))

    print("Average accuracy of validation sets: {0:.3f}".format(np.mean(precisions) * 100))
    print("Average recall rate of validation sets： {0:.3f}".format(np.mean(recalls)))
    m0.train()
    return np.mean(precisions) * 100


def train(args):
    logging.basicConfig(filename=os.path.join(args.data, "training.log"), level=logging.INFO)
    train_loader, validate_loader = load_data(args)
    vocal_size = max(train_loader.maxID, validate_loader.maxID) + 8
    print("vocal_size= {}".format(vocal_size))
    epoch = validate_loader.size // constants.val_batch
    acmc_pairs, acmc_groups, acmc_maps = [], [], []
    for ii in range(epoch):
        validate_batch = validate_loader.all_trjs[ii * constants.val_batch:(ii + 1) * constants.val_batch]
        greedy = ACMC()
        pairs, groups, trj_map_group = greedy.get_groups(validate_batch)
        acmc_pairs.append(pairs)  # pair = [id1, id2, common_values]
        acmc_groups.append(groups)
        acmc_maps.append(trj_map_group)

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

            a1, p1, n1 = train_loader.get_apn_cross()
            if len(a) != 0:
                diff, t_loss1 = get_triplet_loss(a1, p1, n1, m0, triplet_loss, args)
                if iteration % args.print_freq == 0:
                    print("The number of apn:{0}  difference ： {1}".format(len(a1.src[0]), diff))

            a2, p2, n2 = train_loader.get_inner_apn()
            if len(a2) != 0:
                diff, inner_loss = get_triplet_loss(a2, p2, n2, m0, triplet_loss, args)
                if iteration % args.print_freq == 0:
                    print("The number of apn:{0}  difference ： {1}".format(len(a2.src[0]), diff))
            a3, p3, n3 = train_loader.get_common_apn()
            if len(a3) != 0:
                diff, common_loss = get_triplet_loss(a3, p3, n3, m0, triplet_loss, args)
                if iteration % args.print_freq == 0:
                    print("The number of apn:{0}  difference ： {1}".format(len(a3.src[0]), diff))
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
                    recall = validate(validate_loader, m0, args, acmc_groups)
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

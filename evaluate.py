import torch
import os
from trainer.models import EncoderDecoder
from loader.data_utils import pad_arrays_keep_invp
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import numpy as np
import time
from baselines.GS_ACMC import ACMC
from baselines.GS_ECMC import ECMC
import settings
import h5py
from loader.data_loader import DataLoader


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


def t2vec(args, trj_src, vocab_size):
    m0 = EncoderDecoder(args, vocab_size)
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        if torch.cuda.is_available():
            m0.cuda()
        m0.eval()
        vectors = []
        src, lengths, invp = trj_src[0], trj_src[1], trj_src[2]
        if torch.cuda.is_available():
            src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
        h, _ = m0.encoder(src, lengths)  # [Layers*2，Number of trajectories in this group，Number of hidden layers][6，settings.n，128]
        h = m0.encoder_hn2decoder_h0(h)
        h = h.transpose(0, 1).contiguous()
        vectors.append(h[invp].cpu().data)
        vectors = torch.cat(vectors)  # [10,3,256]
        vectors = vectors.transpose(0, 1).contiguous()  # [3,10,256]
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
    return vectors[m0.num_layers-1].tolist()


def clustering(vectors):
    t1 = time.time()
    if settings.c_method == 1:
        c = KMeans(n_clusters=settings.n).fit(vectors)
    elif settings.c_method == 2:
        c = DBSCAN(eps=settings.eps, min_samples=settings.mt).fit(vectors)
    else:
        c = AgglomerativeClustering(n_clusters=settings.n).fit(vectors)
    print('Clustering time：'+str(time.time()-t1))
    return c.labels_


def get_recall(labels, all_groups):
    TP_count = 0
    FN_count = 0
    for g in all_groups:
        for i in range(len(g)):
            if labels[g[i]] == -1:
                continue
            for j in range(i+1, len(g)):
                if labels[g[j]] == -1:
                    continue
                if labels[g[i]] == labels[g[j]]:
                    TP_count += 1
                else:
                    FN_count += 1
    recall = TP_count*100/(TP_count+FN_count)
    return recall
    

# Gets all the original traces of the test set
def get_raw_trajectory():
    # Get the original track data of all validation sets
    trajectories = []
    train_ratio = settings.train_ratio
    with h5py.File("E:/data/beijing.h5", 'r') as f:
        trj_nums = f.attrs['num']
    train_num = int(train_ratio * trj_nums)
    with h5py.File("E:/data/beijing.h5", 'r') as f:
        for i in range(train_num, trj_nums):
            trip = np.array(f.get('trips/' + str(i + 1)))
            ts = np.array(f.get('timestamps/' + str(i + 1)))
            trajectory = []
            time_span = 86400 // settings.time_size
            for (lon, lat), t in zip(trip, ts):
                trajectory.append([lon, lat, int(t) // time_span])
            trajectories.append(trajectory)
    return trajectories


def evaluate(args):
    ''' 1. Get a batch of trajectories
        1.1 Exact group t2G1 of the batch track is obtained
        1.2 Encodeing this group of trajectories
    '''
    # Gets the validation set original trajectory for ECMC
    trajectories = get_raw_trajectory()[0:args.max_val_nums]
    # Obtain all trajectory token sequences of the test set for ACMC, t2vec
    train_loader = DataLoader(os.path.join(args.data, "train"), args.batch)
    train_loader.load(args.max_train_num)
    validate_loader = DataLoader(os.path.join(args.data, "val"), args.batch)
    validate_loader.load(args.max_val_nums)
    vocal_size = max(train_loader.maxID, validate_loader.maxID) + 8
    print("vocal_size= {}".format(vocal_size))

    # ECMC ACMC groups are calculated by batch
    epoch = len(trajectories) // settings.val_batch
    print(epoch)
    for ii in range(epoch):
        # Read a batch of validation set trajectories
        t1 = time.time()
        validate_batch = trajectories
        # print(len(validate_batch[0]), validate_batch[0])
        # print(len(validate_batch[-1]), validate_batch[-1])
        e = ECMC()
        all_pairs, all_groups, trj_map_group = e.get_groups(validate_batch)
        print(all_groups)
        print("ECMC time : ", time.time() - t1)

        t1 = time.time()
        validate_batch = validate_loader.all_trjs
        # Get companion Pairs, companion Groups, companion trj2group in the batch trajectory
        greedy = ACMC()
        pairs, groups, trj_map_group = greedy.get_groups(validate_batch)
        print(groups)
        print("ACMC time: ", time.time() - t1)
        print("**********")

        t1 = time.time()
        vectors = t2vec(args, pad_arrays_keep_invp(validate_batch), vocal_size)
        labels = clustering(vectors)
        recall = get_recall(labels, all_groups)  # recall rate after cluster
        print("recall rate: ", recall)
        print("Model time : ", time.time() - t1)

        print("\n\n")


def get_hit_number(args, trj_number):
    ''' 1. Get a batch of trajectories
        1.1 Exact group t2G1 of the batch track is obtained
        1.2 Encodeing this group of trajectories
    '''
    # Gets the validation set original trajectory for ECMC
    trajectories = get_raw_trajectory()[0:trj_number]
    # Obtain all trajectory token sequences of the test set for ACMC, t2vec
    train_loader = DataLoader(os.path.join(args.data, "train"), args.batch)
    train_loader.load(args.max_train_num)
    validate_loader = DataLoader(os.path.join(args.data, "val"), args.batch)
    validate_loader.load(args.max_val_nums)
    validate_loader.all_trjs = validate_loader.all_trjs[0:trj_number]
    vocal_size = max(train_loader.maxID, validate_loader.maxID) + 8
    print("vocal_size= {}".format(vocal_size))


    t1 = time.time()
    validate_batch = trajectories
    e = ECMC()
    all_pairs, all_groups, trj_map_group = e.get_groups(validate_batch)
    hit_numbers = [len(item)*(len(item)-1)/2 for item in all_groups]
    print("hitnum: ", np.sum(hit_numbers))
    print("ECMC time: ", time.time() - t1)

    t1 = time.time()
    validate_batch = validate_loader.all_trjs
    # Get companion Pairs, companion Groups, companion trj2group in the batch trajectory
    greedy = ACMC()
    pairs, groups, trj_map_group = greedy.get_groups(validate_batch)
    hit_numbers = [len(item) * (len(item)-1) / 2 for item in groups]
    print("hitnum: ", np.sum(hit_numbers))
    print("ACMC time: ", time.time() - t1)
    print("**********")

    t1 = time.time()
    vectors = t2vec(args, pad_arrays_keep_invp(validate_batch), vocal_size)
    labels = clustering(vectors)
    recall = get_recall(labels, all_groups)  # recall rate after cluster
    print("recall rate: ", recall)
    print("Model time : ", time.time() - t1)
    grouped_ids = sum(all_groups, [])
    co_pair_num = 0
    al_pair_num = 0
    for i in range(settings.n):
        # Extract the trajectory in each cluster
        cluster = np.where(labels == i)[0].tolist()
        trj_cluster = np.array(validate_batch)[cluster].tolist()
        for mm in range(len(trj_cluster)):
            for jj in range(mm + 1, len(trj_cluster)):
                # if cluster[jj] not in grouped_ids:
                #     continue
                if len(common_values(trj_cluster[mm], trj_cluster[jj])) >= settings.min_lifetime:
                    co_pair_num += 1
                al_pair_num += 1
    print(co_pair_num, co_pair_num/al_pair_num)


def period_test(args, trj_number, index):
    ''' 1. Get a batch of trajectories
        1.1 Exact group t2G1 of the batch track is obtained
        1.2 Encodeing this group of trajectories
    '''
    # Gets the validation set original trajectory for ECMC
    trajectories = get_raw_trajectory()
    trj_number = min(trj_number, len(trajectories))
    # Obtain all trajectory token sequences of the test set for ACMC, t2vec
    train_loader = DataLoader(os.path.join(args.data, "train"), args.batch)
    train_loader.load(args.max_train_num)
    validate_loader = DataLoader(os.path.join(args.data, "val"), args.batch)
    validate_loader.load(trj_number)
    trj_number = min(trj_number, len(validate_loader.all_trjs))
    trajectories = trajectories[0:trj_number]
    validate_loader.all_trjs = validate_loader.all_trjs[0:trj_number]
    vocal_size = max(train_loader.maxID, validate_loader.maxID) + 8

    peak, work, casual = [], [], []
    for ii in range(trj_number):
        trj = trajectories[ii]
        time1 = trj[0][2] * (86400/settings.time_size) / 3600
        time2 = trj[-1][2] * (86400/settings.time_size) / 3600
        if 0 <= time1 <= 6:
            casual.append(ii)
        elif 10 <= time1 <= 17 and 10 <= time2 <= 17:
            work.append(ii)
        else:
            peak.append(ii)
    print("peak work casual size: ", len(peak), len(work), len(casual))
    print("vocal_size= {}".format(vocal_size))

    if index == 1:
        validate_batch = np.array(trajectories)[peak].tolist()
        # validate_batch = np.narray(validate_loader.all_trjs)[peak].tolist()
    elif index == 2:
        validate_batch = np.array(trajectories)[work].tolist()
        # validate_batch = np.narray(validate_loader.all_trjs)[work].tolist()
    else:
        validate_batch = np.array(trajectories)[casual].tolist()

    t1 = time.time()

    e = ECMC()
    all_pairs, all_groups, trj_map_group = e.get_groups(validate_batch)
    hit_numbers = [len(item)*(len(item)-1)/2 for item in all_groups]
    print(len(all_pairs))
    print("hitnum: ", np.sum(hit_numbers))
    print("ECMC time: ", time.time() - t1)

    t1 = time.time()
    if index == 1:
        validate_batch = np.array(validate_loader.all_trjs)[peak].tolist()
    elif index == 2:
        validate_batch = np.array(validate_loader.all_trjs)[work].tolist()
    else:
        validate_batch = np.array(validate_loader.all_trjs)[casual].tolist()
    # Get companion Pairs, companion Groups, companion trj2group in the batch trajectory
    greedy = ACMC()
    pairs, groups, trj_map_group = greedy.get_groups(validate_batch)
    hit_numbers = [len(item) * (len(item)-1) / 2 for item in groups]
    print(len(pairs))
    print("hitnum: ", np.sum(hit_numbers))
    print("ACMC time : ", time.time() - t1)
    print("**********")

    t1 = time.time()
    vectors = t2vec(args, pad_arrays_keep_invp(validate_batch), vocal_size)
    labels = clustering(vectors)
    # recall = get_recall(labels, all_groups)  # recall rate after cluster
    # print("recall rate: ", recall)
    print("Model time: ", time.time() - t1)
    co_pair_num = 0
    al_pair_num = 0

    pair = 0
    for i in range(settings.n):
        # Extract the track in each cluster
        cluster = np.where(labels == i)[0].tolist()
        pair += len(cluster)*(len(cluster)-1)/2
        trj_cluster = np.array(validate_batch)[cluster].tolist()
        for mm in range(len(trj_cluster)):
            for jj in range(mm + 1, len(trj_cluster)):
                if len(common_values(trj_cluster[mm], trj_cluster[jj])) >= settings.min_lifetime:
                    co_pair_num += 1
                al_pair_num += 1
    print(pair, co_pair_num, co_pair_num/al_pair_num)


if __name__ == "__main__":
    args = settings.set_args()
    # get_hit_number(args, 5000)
    period_test(args, 200000, 3)
    # 10739 4683 4578 beijing
    #

# porto 1500 (300, 350, 400, 450)
# porto (500, 1000, 1500, 2000) 400
# beijing 1000 (100, 200, 300, 400)
# beijing (500 1000 1500 2000) 300
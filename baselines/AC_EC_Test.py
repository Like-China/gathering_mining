"""
Compare the performance of ECMC and ACMC on the test set
"""
import numpy as np
import settings as constants
import settings as settings
import h5py
from baselines.GS_ACMC import ACMC
from baselines.GS_ECMC import ECMC
from settings import set_args
from loader.data_loader import DataLoader
import os
import time


if __name__ == "__main__":
    trajectories = []
    train_ratio = settings.train_ratio
    with h5py.File("F:/data/beijing.h5", 'r') as f:
        trj_nums = f.attrs['num']
    train_num = int(train_ratio * trj_nums)
    val_num = trj_nums - train_num
    with h5py.File("F:/data/beijing.h5", 'r') as f:
        for i in range(train_num, trj_nums):
            trip = np.array(f.get('trips/' + str(i + 1)))
            ts = np.array(f.get('timestamps/' + str(i + 1)))
            trajectory = []
            time_span = 86400 // settings.time_size
            for (lon, lat), t in zip(trip, ts):
                trajectory.append([lon, lat, int(t)//time_span])
            trajectories.append(trajectory)

    # ACMC
    args = set_args()
    validate_loader = DataLoader(os.path.join("../data/beijing/beijing500200/val"), args.batch)
    validate_loader.load(args.max_val_nums)
    epoch = len(trajectories) // constants.val_batch
    for ii in range(epoch):
        t1 = time.time()
        validate_batch = trajectories[ii * constants.val_batch:(ii + 1) * constants.val_batch]
        print(len(validate_batch[0]), validate_batch[0])
        print(len(validate_batch[-1]), validate_batch[-1])
        e = ECMC()
        all_pairs, all_groups, trj_map_group = e.get_groups(validate_batch)
        print(all_groups)
        print("ECMC time : ", time.time()-t1)

        t1 = time.time()
        validate_batch = validate_loader.all_trjs[ii * constants.val_batch:(ii + 1) * constants.val_batch]
        if len(validate_batch) > 0 :
            print(len(validate_batch[0]), validate_batch[0])
            print(len(validate_batch[-1]), validate_batch[-1])
        greedy = ACMC()
        pairs, groups, trj_map_group = greedy.get_groups(validate_batch)
        print(groups)
        print("ACMC time: ", time.time() - t1)
        print("**********")

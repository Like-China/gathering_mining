from loader.data_utils import *
from tqdm import tqdm
import settings as constants
from settings import set_args


class DataLoader:
    """
    Data loading class
    The training set loads data disordered, and the test set loads data orderly
    """
    def __init__(self, srcfile, batch_size):
        self.srcfile = srcfile
        self.batch_size = batch_size
        self.maxID = 0
        self.minID = 100
        self.all_trjs = []
        self.size = 0

    def load(self, max_num):
        stream = open(self.srcfile, 'r')
        num_line = 0
        with tqdm(total=max_num, desc='Read trajectory file', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            for s in stream:
                s = [int(x) + constants.START for x in s.split()]
                if len(s) > 0:
                    self.all_trjs.append(s)
                    num_line += 1
                    # Record the maximum and minimum token values to verify whether there are conversion errors
                    if max(s) > self.maxID:
                        self.maxID = max(s)
                    if min(s) < self.minID:
                        self.minID = min(s)
                pbar.update(1)
                if num_line >= max_num:
                    break
            stream.close()
        self.size = len(self.all_trjs)

    def get_batch(self):
        rand_i = np.random.randint(0, self.size - self.batch_size-1)
        # if self.batch_size < self.size:
        #     rand_index = np.random.randint(0, self.size, self.batch_size)
        # else:
        #     rand_index = np.random.randint(0, self.size, self.size)
        return np.array(self.all_trjs)[rand_i: rand_i+self.batch_size].tolist()

    '''
        Three track sets a, p and n are generated after removing points in a batch of tracks with a certain probability
        a, p, n are all taken from the same trajectory; The overlap degree of sub-track of a and p sampling is higher than that of pattern track of a and n sampling, that is, a and p are more similar
    '''
    def get_inner_apn(self):
        a_src, p_src, n_src = [], [], []
        args = set_args()
        selected_trj_ids = np.random.choice(len(self.all_trjs), self.batch_size).tolist()
        trgs = np.array(self.all_trjs)[selected_trj_ids].tolist()
        for i in range(len(trgs)):
            trg = np.array(trgs[i])
            if len(trg) < 5:
                continue
            a1, a3, a5 = 0, len(trg) // 2, len(trg)
            a2, a4 = (a1 + a3) // 2, (a3 + a5) // 2
            rate = np.random.choice([0.3, 0.4, 0.6])
            if np.random.rand() > 0.5:
                a_src.append(random_subseq(trg[a1:a4], rate))
                p_src.append(random_subseq(trg[a2:a5], rate))
                n_src.append(random_subseq(trg[a3:a5], rate))
            else:
                a_src.append(random_subseq(trg[a2:a5], rate))
                p_src.append(random_subseq(trg[a1:a4], rate))
                n_src.append(random_subseq(trg[a1:a3], rate))
            if len(a_src) > args.max_apn_num:
                break
        a = pad_arrays_pair(a_src)
        p = pad_arrays_pair(p_src)
        n = pad_arrays_pair(n_src)
        return a, p, n

    '''
        Three sets of tracks are generated after removing points in a batch of tracks with a certain probability: a, p, n
    '''
    def get_common_apn(self):
        args = set_args()
        a_src, p_src, n_src = [], [], []
        selected_trj_ids = np.random.choice(len(self.all_trjs), self.batch_size).tolist()
        trgs = np.array(self.all_trjs)[selected_trj_ids].tolist()
        for i in range(len(trgs)):
            trg = np.array(trgs[i])
            if len(trg) < 5:
                continue
            rate = np.random.choice([0.3, 0.4, 0.6])
            sub1 = random_subseq(trg, rate)
            sub2 = random_subseq(trg, rate)
            sub3 = random_subseq(trg, rate)
            # 经测试下面函数比自己写的函数快
            common_num1 = list(set(sub1).intersection(set(sub2)))
            common_num2 = list(set(sub1).intersection(set(sub3)))

            if len(common_num1) < len(common_num2):
                sub2, sub3 = sub3, sub2

            a_src.append(sub1)
            p_src.append(sub2)
            n_src.append(sub3)

            if len(a_src) > args.max_apn_num:
                break

        a = pad_arrays_pair(a_src)
        p = pad_arrays_pair(p_src)
        n = pad_arrays_pair(n_src)
        return a, p, n

    def get_apn_cross(self):
        """
        Get three batch number of track sets, a, p, n
        The center of the trajectory in a is closer to the trajectory in p
        """

        a_src = self.get_batch()[0: 100]
        p_src = self.get_batch()[0: 100]
        n_src = self.get_batch()[0: 100]

        for i in range(len(a_src)):
            common_num1 = list(set(a_src[i]).intersection(set(p_src[i])))
            common_num2 = list(set(a_src[i]).intersection(set(n_src[i])))
            if len(common_num1) < len(common_num2):
                p_src[i], n_src[i] = n_src[i], p_src[i]

        a = pad_arrays_pair(a_src)
        p = pad_arrays_pair(p_src)
        n = pad_arrays_pair(n_src)
        return a, p, n

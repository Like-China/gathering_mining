import numpy as np
import torch
import settings as constants
from collections import namedtuple
from tqdm import tqdm
import time

def argsort(seq):
    """
    sort by length in reverse order
    ---
    seq (list[array[int32]])
    Returns a set of ids by ordering a set of sentences in order of sequence length
    such as src=[[1,2,3],[3,4,5,6],[2,3,4,56,3]] , return 2,1,0
    """
    return [x for x,y in sorted(enumerate(seq),
                                key = lambda x: len(x[1]),
                                reverse=True)]


def pad_array(a, max_length, PAD=constants.PAD):
    """
    a (array[int32])
    In single track zeroing operation, the length is the maximum length of the batch track
    [1,2,3] -> [1,2,3,0,0,..]
    """
    return np.concatenate((a, [PAD]*(max_length - len(a))))


def pad_arrays(a):
    """
    a array(array[int32])
    In the operation of adding zeros for multiple tracks, the length of adding zeros for each track is the same as that of the batch longest track
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
    
    1. ero is added to the track to make the length of all tracks the same
    2. Sort the track length from largest to smallest
    3. Return the TD class where the list of track points is transposed, with each column representing a track
    4. return format['src', 'lengths', 'invp']
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
    Enter p and return invp, the index of the value for each position of p
    idx = [5, 7, 8, 9, 6, 1, 2, 0, 3, 4]
    invp(idx) = [7, 5, 6, 8, 9, 0, 4, 1, 2, 3]  
    invp[p[i]] = i; If there is 45 in p and I want to know where 45 is in P, then INVp [45] will tell us the answer
    invp[i] = p.index(i)
    """
    p = np.asarray(p)
    invp = np.empty_like(p)
    for i in range(p.size):
        invp[p[i]] = i
    return invp


def random_subseq(a, rate):
    """
    Remove the points in a[1:-2] with probability
    Dropping some points between a[3:-2] randomly according to rate.

    Input:
    a (array[int])
    rate (float)
    """
    idx = np.random.rand(len(a)) < rate
    idx[0], idx[-1] = True, True
    return a[idx]


def pad_arrays_keep_invp(src):
    """
    Pad arrays and return inverse permutation
    Input:
    src (list[array[int32]])
    ---
    Output:
    src (seq_len, batch)
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    """
    idx = argsort(src)  # [5, 7, 8, 9, 6, 1, 2, 0, 3, 4]
    src = list(np.array(src)[idx])
    lengths = list(map(len, src))  # [13, 13, 12, 12, 10, 5, 5, 4, 4, 3]
    lengths = torch.LongTensor(lengths)  
    src = pad_arrays(src) 
    invp = torch.LongTensor(invpermute(idx))  # [7, 5, 6, 8, 9, 0, 4, 1, 2, 3]
    return src.t().contiguous(), lengths.view(1, -1), invp

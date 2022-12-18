from rtree import index
from loader.data_loader import Loader
from tqdm import tqdm


def my_rtree():
    from rtree import index
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    idx = index.Index('3d_index', properties=p, interleaved=False, overwrite=True)
    return idx


if __name__ == "__main__":
    idx = my_rtree()
    # 加载数据
    trajectory_set = Loader().load(100000)
    # 建立索引
    for trj in trajectory_set:
        idx.insert(trj.id, trj.mbr)

    # 索引查询
    count = []
    for trj in tqdm(trajectory_set):
        interset_trjs = list(idx.intersection(trj.mbr))
        count.append(len(interset_trjs))


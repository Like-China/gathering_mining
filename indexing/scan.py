import networkx as nx
import random
import math
import matplotlib.pyplot as plt
import numpy as np


class SCAN:

    def __init__(self, G, epsilon=0.5, mu=3):
        self._G = G
        self._epsilon = epsilon
        self._mu = mu

    # 节点的 ϵ-邻居定义为与其相似度不小于 ϵ 的节点所组成的集合
    def get_epsilon_neighbor(self, node):
        return [neighbor for neighbor in self._G.neighbors(node) if
                cal_similarity(self._G, node, neighbor) >= self._epsilon]

    # 判断是否是核节点
    def is_core(self, node):
        # 核节点是指ϵ-邻居的数目大于 μ 的节点。
        return len(self.get_epsilon_neighbor(node)) >= self._mu

    # 获得桥节点和离群点
    def get_hubs_outliers(self, communities):
        other_nodes = set(list(self._G.nodes()))
        node_community = {}
        for i, c in enumerate(communities):
            for node in c:
                # 已经有社区的节点删除
                other_nodes.discard(node)
                # 为节点打上社区标签
                node_community[node] = i
        hubs = []
        outliers = []
        # 遍历还未被划分到社区中的节点
        for node in other_nodes:
            neighbors = self._G.neighbors(node)
            # 统计该节点的邻居节点所在的社区  大于1为桥节点 否则为离群点
            neighbor_community = set()
            for neighbor in neighbors:
                if neighbor in node_community:
                    neighbor_community.add(node_community[neighbor])
            if len(neighbor_community) > 1:
                hubs.append(node)
            else:
                outliers.append(node)
        return hubs, outliers

    def execute(self):
        # 随机访问节点
        visit_sequence = list(self._G.nodes())
        random.shuffle(visit_sequence)
        communities = []
        for node_name in visit_sequence:
            node = self._G.nodes[node_name]
            # 如果节点已经分类好 则迭代下一个节点
            if node.get("classified"):
                continue
            # 如果是核节点 则是一个新社区
            if self.is_core(node_name):  # a new community
                community = [node_name]
                communities.append(community)
                node["type"] = "core"
                node["classified"] = True
                # 获得该核心点的ϵ-邻居
                queue = self.get_epsilon_neighbor(node_name)
                # 首先将核心点v的所有其 ϵ-邻居放进队列中。对于队列中的每个顶点，它计算所有直接可达的顶点，并将那些仍未分类的顶点插入队列中。重复此操作，直到队列为空
                while len(queue) != 0:
                    temp = queue.pop(0)
                    # 若该ϵ-邻居没被分类 则将它标记为已分类 并添加到该社区
                    if not self._G.nodes[temp].get("classified"):
                        self._G.nodes[temp]["classified"] = True
                        community.append(temp)
                    # 如果该点不是核心节点 遍历下一个节点 否则继续(不是核心节点则说明可达的点到该点终止了)
                    if not self.is_core(temp):
                        continue
                    # 如果是核心节点 获得他的ϵ-邻居 看他的ϵ-邻居是否有还未被划分的 添加到当前社区
                    R = self.get_epsilon_neighbor(temp)
                    for r in R:
                        node_r = self._G.nodes[r]
                        is_classified = node_r.get("classified")
                        if is_classified:
                            continue
                        node_r["classified"] = True
                        community.append(r)
                        # r是核心节点还能可达其它节点 还没观察他的ϵ-邻居  放入queue中
                        queue.append(r)
        return communities


def cal_similarity(G, node_i, node_j):
    # 按照公式计算相似度
    # 节点相似度定义为两个节点共同邻居的数目与两个节点邻居数目的几何平均数的比值（这里的邻居均包含节点自身）
    s1 = set(G.neighbors(node_i))
    s1.add(node_i)
    s2 = set(G.neighbors(node_j))
    s2.add(node_j)
    return len(s1 & s2) / math.sqrt(len(s1) * len(s2))


def draw_spring(G, pos, com):
    """
    G:图
    com：划分好的社区
    node_size表示节点大小
    node_color表示节点颜色
    node_shape表示节点形状
    with_labels=True表示节点是否带标签
    """
    pos = pos  # 节点的布局为spring型
    NodeId = list(G.nodes())
    node_size = [G.degree(i) ** 1.2 * 90 for i in NodeId]  # 节点大小

    plt.figure(figsize=(100, 100))  # 图片大小
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color='w', node_shape='.')

    color_list = ['pink', 'orange', 'r', 'g', 'b', 'y', 'm', 'gray', 'black', 'c', 'brown']*len(com)
    # node_shape = ['s','o','H','D']

    for i in range(len(com)):
        nx.draw_networkx_nodes(G, pos, nodelist=com[i], node_color=color_list[i])
    plt.show()


# 加载图数据集
def load_graph(path):
    G = nx.Graph()
    with open(path, 'r') as text:
        for line in text:
            vertices = line.strip().split(' ')
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target)
    return G


def get_communities(pairs, mu, epsilon):
    G = nx.Graph()
    for line in pairs:
        source, target = line[0], line[1]
        G.add_edge(source, target)
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, with_labels=True, font_weight='bold')
    # plt.show()
    algorithm = SCAN(G, epsilon, mu)
    communities = algorithm.execute()
    hubs_outliers = algorithm.get_hubs_outliers(communities)
    # draw_spring(G,pos, communities)
    return communities, hubs_outliers[0], hubs_outliers[1]




import numpy as np
from examples.text_analyse import text_embedding

def row_normalize(mx):
    """Row-normalize matrix"""
    rowsum = np.array(mx.sum(1))                # 每一行求和
    r_inv = np.power(rowsum, -0.5).flatten()    # 返回一个一维数组
    r_inv[np.isinf(r_inv)] = 0.                 # 一维数组中，如果有inf，变为0
    r_mat_inv = np.eye(r_inv.shape[0])
    for i in range(r_inv.shape[0]):
        r_mat_inv[i][i] = r_inv[i]
    mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)   # 行归一化
    return mx


def load_data_1(dataset='cora'):
    path = r'data_1\{}'.format(dataset)

    feature = []
    with open(path+'.feature', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # print(line)
            feature.append([int(k) for k in line.strip().split()])
            # print(feature)
            # break
    feature_arr = np.array(feature)
    print(feature_arr.shape)

    label = []
    with open(path+'.label', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # print(line)
            label.append([int(line.strip())])
            # print(feature)
            # break
    label_arr = np.array(label)
    print(label_arr.shape)

    adj = np.zeros((label_arr.shape[0], label_arr.shape[0]), dtype=np.int8)
    with open(path+'.edge', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            nodes = [int(k) for k in line.strip().split()]
            # print(nodes)
            adj[nodes[0], nodes[1]] = 1
    print(adj, adj.shape)

    return feature_arr, label_arr, adj


def load_data_2(dataset='cora'):
    path = r'data_2\{}'.format(dataset)

    text = []
    with open(path+'.text', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            text.append(line.strip().split(maxsplit=1)[1])
    feature_arr = text_embedding(text)
    print(feature_arr.shape)
    # exit(0)

    label = []
    with open(path+'.label', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # print(line)
            label.append([int(line.strip().split()[1])])
            # print(feature)
            # break
    label_arr = np.array(label)
    print(label_arr.shape)

    adj = np.zeros((label_arr.shape[0], label_arr.shape[0]), dtype=np.int8)
    with open(path+'.edge', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            nodes = [int(k) for k in line.strip().split()]
            # print(nodes)
            adj[nodes[0], nodes[1]] = 1
    print(adj, adj.shape)

    return feature_arr, label_arr, adj


if __name__ == "__main__":
    # load_data_1()
    load_data_2()
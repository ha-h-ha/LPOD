import pandas as pd
import numpy as np
from scipy.spatial import KDTree

def LPOD(data,k,lambda_):
    m, n = data.shape
    # 创建一个数组来存储每个观测的异常值分数
    scores = np.zeros(m)
    # 创建一个KDTree
    tree = KDTree(data)
    # 对于每个观测
    for i in range(m):
        x = data[i,:]
        # 找到k个最近邻居
        dists, inds = tree.query(x, k=k)
        # 计算N
        N = data[inds[1:], :]
        # 计算N的低秩近似
        N_bar = low_rank_approx(N, lambda_)
        # 计算N的核范数
        nuclear_norm = np.sum(np.linalg.svd(N_bar, compute_uv=False))
        scores[i] = nuclear_norm
    return scores

def low_rank_approx(D, lambda_):
        # 进行奇异值分解
        U, S, V = np.linalg.svd(D, full_matrices=False)
        r = len(S)
        S_diag = np.diag(S)
        # 低秩近似
        for i in range(r):
            S_diag[i, i] = max(S[i] - lambda_, 0)
        D_bar = np.dot(U, np.dot(S_diag, V))

        return D_bar

if __name__ == '__main__':
    # 加载数据
    df = pd.read_csv('./data_banknote_authentication.txt', sep=',', header=None)
    df=df.values
    X=df[:,0:4]

    # 计算异常值分数
    k=10
    lambda_=0.1
    scores = LPOD(X,k,lambda_)
    print(scores)

    # PR
    Y=df[:,4]
    outliers = np.argsort(scores)[::-1][:100]
    print('准确度：', np.sum(Y[outliers] == 1) / 100)


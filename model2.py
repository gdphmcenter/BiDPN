import argparse

import numpy as np
import torch
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from model1 import *
import os
from pathlib import Path
from clusterone import get_data_path, get_logs_path
from dataset import Mydataset

from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

# coding=utf8

use_cuda = True
latent_size = 4500
batch_size = 3120
model_path = './model/netE_epoch_749.pth'

train_loader = torch.utils.data.DataLoader(
    Mydataset(False),
    batch_size=batch_size, shuffle=False)


def tocuda(x):
    if use_cuda:
        return x.cuda()
    return x


def save_feature(data, label):
    np.save("./feature/SCL_feature_150_3_e749.npy", data)
    np.save("./feature/SCL_feature_labels_150_3_e749.npy", label)


netE = tocuda(Encoder(latent_size, False))
netE.load_state_dict(torch.load(model_path))

n_components = 2  # 绘制三维图形还是二维图形
for (data, target) in train_loader:
    data = tocuda(data)
    target = target.cpu().detach().numpy()
    X = netE(data).squeeze(2).squeeze(2)
    X = X.cpu().detach().numpy()

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y = tsne.fit_transform(X)  # 转换后的输出
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时

    if n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 绘制散点图
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=target)
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


        def on_move(event):
            ax.view_init(elev=event.ydata, azim=event.xdata)


        fig.canvas.mpl_connect('motion_notify_event', on_move)
        # 显示图形
        plt.show()
    elif n_components == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # 绘制散点图
        ax.scatter(Y[:, 0], Y[:, 1], c=target)
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # 显示图形
        plt.show()

    save_feature(X, target)

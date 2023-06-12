
import torch
import time
import os
import numpy as np
import torch.optim as optim
import torch.nn as nn
from Data_Sampler import Sampler
from Model_Fusion_Network import Fusion_Network
from Model_Fusion_Loss import Fusion_Loss
# from tensorboardX import SummaryWriter
from Parser import Parser
from data import Dataloader, Singal, SingalBearing, SingalGearbox, SingalSnsHVCM
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_tsne(x, y, labels, title):
    """
    :param x: (numpy array): feature embeddings
    :param y: (numpy array): ground truth labels
    :param labels: (list): label names
    :param title: (str): plot title
    """
    tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=12, learning_rate=200, n_jobs=-1)
    x_tsne = tsne.fit_transform(x)
    plt.figure(figsize=(10, 10))
    plt.title(title)
    for i, label in enumerate(labels):
        idxs = y == i
        plt.scatter(x_tsne[idxs, 0], x_tsne[idxs, 1], label=label)
    plt.legend()
    plt.show()


def init_manual_seed(opt):
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def init_model():
    model = Fusion_Network()
    return model

def init_optimizer(opt, model):
    optimizer = optim.Adam(params=model.parameters(), lr=opt.learning_rate)
    return optimizer

def init_lr_scheduler(opt, optimizer):
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=opt.lr_scheduler_gamma, step_size=opt.lr_scheduler_step)
    return lr_scheduler

def train(opt, model, data_loader, optimizer, lr_scheduler):
    best_acc = 0
    best_test_acc = 0
    loss_fn = Fusion_Loss(opt=opt)
    for epoch in range(1, 1+opt.epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        test_loss = 0
        val_acc = 0
        test_acc = 0
        # Model train
        model.train()
        for t_eposide in range(1, 1+opt.episodes_per_epoch):
            optimizer.zero_grad()
            [DE_support,
             DE_query] = data_loader['train'].get_task_batch(num_ways=opt.k_train,
                                                             num_shots=opt.n_train,
                                                             num_queries=opt.q_train,
                                                             seed=t_eposide + epoch * opt.episodes_per_epoch,
                                                             emb_size=4500)
            DE_query_output, DE_prototype= model(DE_support=DE_support, DE_query=DE_query)
            t_loss, t_acc = loss_fn.forward(DE_query_output=DE_query_output, DE_prototype=DE_prototype)
            t_loss.backward()
            optimizer.step()
            train_loss += t_loss.item()
            train_acc += t_acc.item()
            torch.cuda.empty_cache()
        lr_scheduler.step()
        train_loss = train_loss / opt.episodes_per_epoch
        train_acc = train_acc / opt.episodes_per_epoch

        X_test = []
        y_test = []
        y_test = []
        y_pred = []
        # Model eval
        if epoch%1 == 0:
            print('======In Epoch {}======'.format(epoch))
            print('Train loss is {}, train accuracy is {}'.format(train_loss, train_acc))
            model.eval()
            with torch.no_grad():
                for v_episode in range(1, 1+opt.episodes_per_epoch):
                    [v_DE_support,
                     v_DE_query] = data_loader['val'].get_task_batch(num_ways=opt.k_val,
                                                                     num_shots=opt.n_val,
                                                                     num_queries=opt.q_val,
                                                                     seed=v_episode + epoch * opt.episodes_per_epoch,
                                                                     emb_size = 4500)
                    v_DE_query_output, v_DE_prototype = model(DE_support=v_DE_support, DE_query=v_DE_query)
                    v_loss, v_acc = loss_fn.forward(DE_query_output=v_DE_query_output, DE_prototype=v_DE_prototype)
                    val_loss += v_loss.item()
                    val_acc += v_acc.item()
                val_loss = val_loss / opt.episodes_per_epoch
                val_acc = val_acc / opt.episodes_per_epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                print('Val loss is {}, val accuracy is {}best acc{}'.format(val_loss, val_acc, best_acc))

                # for v_episode in range(1, 1 + opt.episodes_per_epoch):
                #     [v_DE_support,
                #      v_DE_query] = data_loader['test'].get_task_batch(num_ways=opt.k_val,
                #                                                       num_shots=opt.n_val,
                #                                                       num_queries=opt.q_val,
                #                                                       seed=v_episode + epoch * opt.episodes_per_epoch,
                #                                                       emb_size=4500)
                #     v_DE_query_output, v_DE_prototype = model(DE_support=v_DE_support, DE_query=v_DE_query)
                #     v_loss, v_acc = loss_fn.forward(DE_query_output=v_DE_query_output, DE_prototype=v_DE_prototype)
                #     test_loss += v_loss.item()
                #     test_acc += v_acc.item()
                # #
                #     # 将预测标签添加到y_pred
                #     y_pred.append(v_DE_query_output.argmax(dim=1).cpu().numpy())
                #
                #     # 将真实标签添加到y_test
                #     [train_data, test_data, val_data, test_label] = SingalSnsHVCM()
                #     y_test.append(test_label[v_episode - 1])
                #
                #     torch.cuda.empty_cache()
                #
                # test_loss = test_loss / opt.episodes_per_epoch
                # test_acc = test_acc / opt.episodes_per_epoch
                # if test_acc > best_test_acc:
                #     best_test_acc = test_acc
                #
                # print('Test loss is {}, test accuracy is {} best acc{}\n'.format(test_loss, test_acc, best_test_acc))
                #
                # # 使用t-SNE将嵌入空间可视化为二维平面
                # tsne = TSNE(n_components=2, random_state=0)
                # y_test_tsne = tsne.fit_transform(y_test)
                #
                # # 绘制测试集的t-SNE图
                # plt.figure(figsize=(10, 10))
                # colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                # for i in range(opt.k_val):
                #     plt.scatter(y_test_tsne[y_pred == i, 0], y_test_tsne[y_pred == i, 1], c=colors[i],
                #                 label='Class {}'.format(i))
                # plt.legend()
                # plt.title('t-SNE Visualization of Test Set')
                # plt.show()


                #原始代码
                for v_episode in range(1, 1 + opt.episodes_per_epoch):
                    [v_DE_support,
                     v_DE_query] = data_loader['test'].get_task_batch(num_ways=opt.k_val,
                                                                     num_shots=opt.n_val,
                                                                     num_queries=opt.q_val,
                                                                     seed=v_episode + epoch * opt.episodes_per_epoch,
                                                                     emb_size=4500)
                    v_DE_query_output, v_DE_prototype = model(DE_support=v_DE_support, DE_query=v_DE_query)
                    v_loss, v_acc = loss_fn.forward(DE_query_output=v_DE_query_output, DE_prototype=v_DE_prototype)
                    test_loss += v_loss.item()
                    test_acc += v_acc.item()
                    torch.cuda.empty_cache()
                test_loss = test_loss / opt.episodes_per_epoch
                test_acc = test_acc / opt.episodes_per_epoch
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                print('Test loss is {}, test accuracy is {} best acc{}\n'.format(test_loss, test_acc, best_test_acc))

def main():
    [train_data, test_data, val_data] = SingalSnsHVCM()
    train_loader = Dataloader(data=train_data)
    test_loader = Dataloader(data=test_data)
    valid_loader = Dataloader(data=val_data)
    data_loader = {'train': train_loader,
                   'val': valid_loader,
                   'test': test_loader
                  }
    option = Parser().parse_args()
    init_manual_seed(opt=option)
    train_model = init_model().cuda()
    optimizer = init_optimizer(opt=option, model=train_model)
    lr_scheduler = init_lr_scheduler(opt=option, optimizer=optimizer)
    train(opt=option, model=train_model, data_loader=data_loader, optimizer=optimizer, lr_scheduler=lr_scheduler)


if __name__ == '__main__':
    main()

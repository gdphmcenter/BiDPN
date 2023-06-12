# -*- coding:utf-8 -*-
"""
author:Ybin Chan
time：2020年12月18日
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def EuclideanDist(xq, prototype):                   #xq：Tensor(k*q, dim), prototype: Tensor(k, dim), k为类别数量，q为query set每类的数量， dim为特征维度
    query_num = xq.size(0)                          #query_num = k*q
    dim = xq.size(1)                                #dim = dim
    prototype_class = prototype.size(0)             #prototype_class = k
    xq = xq.unsqueeze(dim=1).expand(query_num, prototype_class, dim)
    prototype = prototype.unsqueeze(dim=0).expand(query_num, prototype_class, dim)
    Euc_dist = pow(xq-prototype, 2).sum(dim=2)       #Tensor(k*q, k)

    return Euc_dist

class Fusion_Loss(nn.Module):
    def __init__(self, opt):
        super(Fusion_Loss, self).__init__()
        self.TiFre_graph_weight = opt.TiFre_graph_weight
        self.Time_weight = opt.Time_weight
        self.Frequency_weight = opt.Frequency_weight

    def forward(self, DE_query_output, DE_prototype):
        # Setting True label
        k = DE_query_output.size(0)
        q = DE_query_output.size(1)
        # if DE_query_output.size(0) != k or DE_query_output.size(1) != q or FFT_query_output.size(0) != k or FFT_query_output.size(1) != q:
        #     print('===Data exists error===')
        y = torch.arange(k).view(k, 1, 1).expand(k, q, 1).long().cuda()
        # # Time_Frequency_Fusion loss
        # TF_query_output1 = TF_query_output.view(k*q, -1)
        # TF_Eucdist = EuclideanDist(xq=TF_query_output1, prototype=TF_prototype)
        # TF_pred_Eucdist = self.TiFre_graph_weight*F.log_softmax(-TF_Eucdist, dim=1)             #tensor[k*q, k]
        # TF_loss_Eucdist = -TF_pred_Eucdist.view(k, q, k)
        # Time field loss
        DE_query_output1 = DE_query_output.view(k*q, -1)
        DE_Eucdist = EuclideanDist(xq=DE_query_output1, prototype=DE_prototype)
        DE_pred_Eucdist = self.Time_weight*F.log_softmax(-DE_Eucdist, dim=1)                    #tensor[k*q, k]
        DE_loss_Eucdist = -DE_pred_Eucdist.view(k, q, k)
        # Frequency field loss
        # FFT_query_output1 = FFT_query_output.view(k*q, -1)
        # FFT_Eucdist = EuclideanDist(xq=FFT_query_output1, prototype=FFT_prototype)
        # FFT_pred_Eucdist = self.Frequency_weight*F.log_softmax(-FFT_Eucdist, dim=1)             #tensor[k*q, k]
        # FFT_loss_Eucdist = -FFT_pred_Eucdist.view(k, q, k)
        #Three Model Fusion loss
        loss_Eucdist = DE_loss_Eucdist
        loss = torch.gather(loss_Eucdist, dim=2, index=y).squeeze().view(-1).mean()
        pred_Eucdist = DE_pred_Eucdist
        _, pred = torch.max(pred_Eucdist, dim=1)
        pred = pred.view(k*q, 1).view(k, q, 1)
        acc = torch.eq(pred, y).to(torch.float32).squeeze().view(-1).mean()

        return loss, acc







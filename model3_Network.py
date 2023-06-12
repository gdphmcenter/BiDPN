
import torch.nn as nn
import torch
import numpy as np

def conv_block1D(in_channel, out_channel):
    return nn.Sequential(nn.Conv1d(in_channel, out_channel, 3, stride=1, padding=1), nn.BatchNorm1d(out_channel),
                         nn.ReLU(), nn.MaxPool1d(2, 2))

def conv_block2D(in_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1), nn.BatchNorm2d(out_channel),
                         nn.ReLU(), nn.MaxPool2d(2, 2))


class Fusion_Network(nn.Module):
    def __init__(self, indim_1d = 1, hiddim1_1d = 16, hiddim2_1d = 8, indim_2d = 3, outdim_2d = 64):
        super(Fusion_Network, self).__init__()
        #Time-Frequency graph model
        self.TiFre_graph_encoder = nn.Sequential(conv_block2D(in_channel=indim_2d, out_channel=outdim_2d),
                                                 conv_block2D(in_channel=outdim_2d, out_channel=outdim_2d),
                                                 conv_block2D(in_channel=outdim_2d, out_channel=outdim_2d),
                                                 conv_block2D(in_channel=outdim_2d, out_channel=outdim_2d))
        #DE(Time field) model
        self.Time_encoder = nn.Sequential(conv_block1D(in_channel=indim_1d, out_channel=hiddim1_1d),
                                     conv_block1D(in_channel=hiddim1_1d, out_channel=hiddim1_1d),
                                     conv_block1D(in_channel=hiddim1_1d, out_channel=hiddim1_1d),
                                     conv_block1D(in_channel=hiddim1_1d, out_channel=hiddim2_1d),
                                     conv_block1D(in_channel=hiddim2_1d, out_channel=hiddim2_1d),
                                     conv_block1D(in_channel=hiddim2_1d, out_channel=hiddim2_1d))

        #FFT(Frequency field) model
        self.Frequency_encoder = nn.Sequential(conv_block1D(in_channel=indim_1d, out_channel=hiddim1_1d),
                                          conv_block1D(in_channel=hiddim1_1d, out_channel=hiddim1_1d),
                                          conv_block1D(in_channel=hiddim1_1d, out_channel=hiddim2_1d),
                                          conv_block1D(in_channel=hiddim2_1d, out_channel=hiddim2_1d))

    def forward(self,  DE_support, DE_query):
        #Time-Frequency graph(2D) forward
        k = DE_support.size(0)
        n = DE_support.size(1)
        q = DE_query.size(1)
        # print(k)
        # print(n)
        # print(q)

        # if k != TF_query.size(0):
        #     print('Data exists error！')
        # TF_support, TF_query = TF_support.view(k*n, 3, 224, 224), TF_query.view(k*q, 3, 224, 224)
        # TF_support_output = self.TiFre_graph_encoder(TF_support)
        # TF_query_output = self.TiFre_graph_encoder(TF_query)
        # TF_support_output = TF_support_output.view(k*n, 12544).view(k, n, 12544)
        # TF_query_output = TF_query_output.view(k*q, 12544).view(k, q, 12544)        #tensor[k, q, 12544],输出的数据
        # TF_prototype = torch.mean(TF_support_output, dim=1)                         #tensor[k, 12544],输出的数据

        #DE(Time field) forward
        if DE_support.size(0) != k or DE_support.size(1) != n or DE_query.size(1) != q:
            print('Data exists error！')
        # DE_support[25, 1, 512] DE_query[5, 1, 512]
        DE_support, DE_query = DE_support.view(k*n, 1, 4500), DE_query.view(k*q, 1, 4500)

        DE_support_output = self.Time_encoder(DE_support)
        # print(DE_support_output.shape)
        DE_query_output = self.Time_encoder(DE_query)
        # print(DE_query_output.shape)

        DE_support_output = DE_support_output.view(k*n, 560).view(k, n, 560)
        DE_query_output = DE_query_output.view(k*q, 560).view(k, q, 560)            #tensor[k, q, 256],输出的数据
        DE_prototype = torch.mean(DE_support_output, dim=1)                         #tensor[k, 256]，输出的数据










        #FFT(Frequency field) forward
        # if FFT_support.size(0) != k or FFT_support.size(1) != n or FFT_query.size(1) != q:
        #     print('Data exists error！')
        # FFT_support, FFT_query = FFT_support.view(k*n, 1, 256), FFT_query.view(k*q, 1, 256)
        # FFT_support_output = self.Frequency_encoder(FFT_support)
        # FFT_query_output = self.Frequency_encoder(FFT_query)
        # FFT_support_output = FFT_support_output.view(k*n, 128).view(k, n, 128)
        # FFT_query_output = FFT_query_output.view(k*q, 128).view(k, q, 128)          #tensor[k, q, 128],输出的数据
        # FFT_prototype = torch.mean(FFT_support_output, dim=1)                       #tensor[k, 128]输出的数据


        return  DE_query_output, DE_prototype
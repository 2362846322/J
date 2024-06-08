#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from mayavi import mlab
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=1):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False)
        self.value_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False), nn.LeakyReLU(negative_slope=0.2))
        #self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        #proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X N*C
        proj_query = self.query_conv(x).permute(0, 2, 3, 1).reshape(m_batchsize * width, height, -1)
        #proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x N(*W*H)
        proj_key = self.key_conv(x).permute(0, 2, 1, 3).reshape(m_batchsize * width, -1, height)
        energy = torch.bmm(proj_query, proj_key)  # transpose check  B*N*N
        attention = self.softmax(energy)  # BX (N) X (N)
        #proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        #proj_value = self.value_conv(x).permute(0, 2, 1, 3).reshape(m_batchsize * width, -1, height)
        proj_value = x.permute(0, 2, 1, 3).reshape(m_batchsize * width, -1, height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.reshape(m_batchsize, width, C, height).permute(0, 2, 1, 3)

        #out = self.gamma * out + x
        return out*x
class ChannelAttention(nn.Module):  #
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.relu = nn.Sigmoid()  # relu不行

    def forward(self, x):
        avg_in = self.avg_pool(x)
        avg_out = self.fc(avg_in)
        max_in = self.max_pool(x)
        max_out = self.fc(max_in)
        out = avg_out + max_out
        return self.relu(out)


class SpatialAttention(nn.Module):  # 平均值与最大值通道拼接
    def __init__(self, kernel_size=1):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)  # 平均值与最大值通道拼接
        x = self.conv1(x)  # 特征卷积
        return self.relu(x)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature,idx # (batch_size, 2*num_dims, num_points, k)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.Sequential(nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))
        self.bn2 = nn.Sequential(nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))
        self.bn3 = nn.Sequential(nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=0.2))
        self.bn4 = nn.Sequential(nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2))
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False)


        self.ca1 = ChannelAttention(in_planes=64)
        self.ca2 = ChannelAttention(in_planes=64)
        self.ca3 = ChannelAttention(in_planes=128)
        self.ca4 = ChannelAttention(in_planes=256)

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()

        # self.sf1 = Self_Attn(64)
        # self.sf2 = Self_Attn(64)
        # self.sf3 = Self_Attn(128)
        # self.sf4 = Self_Attn(256)

        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):

        batch_size = x.size(0)
        x11, idx = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x11 = self.conv1(x11)
        x12 = x11[:, :, :, 0:10]
        x13 = x11[:, :, :, 0:5]
        x11 = x11.max(dim=-1, keepdim=False)[0]
        x12 = x12.max(dim=-1, keepdim=False)[0]
        x13 = x13.max(dim=-1, keepdim=False)[0]
        x1 = torch.stack((x11, x12, x13), dim=3)
        x1 = self.bn1(x1)
        ca1 = self.ca1(x1)
        x1 = ca1 * x1
        #x1 = self.sf1(x1)
        sa1 = self.sa1(x1)
        x1 = sa1 * x1
        x1 = x1.mean(dim=-1, keepdim=False)

        x11 ,idx= get_graph_feature(x1, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x11 = self.conv2(x11)
        x12 = x11[:, :, :, 0:10]
        x13 = x11[:, :, :, 0:5]

        x11 = x11.max(dim=-1, keepdim=False)[0]
        x12 = x12.max(dim=-1, keepdim=False)[0]
        x13 = x13.max(dim=-1, keepdim=False)[0]
        x2 = torch.stack((x11, x12, x13), dim=3)
        x2 = self.bn2(x2)
        ca2 = self.ca2(x2)
        x2 = ca2 * x2
        #x2 = self.sf2(x2)
        sa2 = self.sa2(x2)
        x2 = sa2 * x2
        x2 = x2.mean(dim=-1, keepdim=False)

        x11 ,idx= get_graph_feature(x2, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x11 = self.conv3(x11)
        x12 = x11[:, :, :, 0:10]
        x13 = x11[:, :, :, 0:5]
        x11 = x11.max(dim=-1, keepdim=False)[0]
        x12 = x12.max(dim=-1, keepdim=False)[0]
        x13 = x13.max(dim=-1, keepdim=False)[0]
        x3 = torch.stack((x11, x12, x13), dim=3)
        x3 = self.bn3(x3)
        ca3 = self.ca3(x3)
        x3 = ca3 * x3
        #x3 = self.sf3(x3)
        sa3 = self.sa3(x3)
        x3 = sa3 * x3
        x3 = x3.mean(dim=-1, keepdim=False)

        x11 ,idx = get_graph_feature(x3, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        idx = idx.view(-1, self.k)
        x11 = self.conv4(x11)
        x12 = x11[:, :, :, 0:10]
        x13 = x11[:, :, :, 0:5]
        x11, x11index = x11.max(dim=-1, keepdim=False)

        x12, x12index = x12.max(dim=-1, keepdim=False)
        x13, x13index = x13.max(dim=-1, keepdim=False)

        x4 = torch.stack((x11, x12, x13), dim=3)
        x4 = self.bn4(x4)
        ca4 = self.ca4(x4)
        x4 = ca4 * x4
        #x4 = self.sf4(x4)
        sa4 = self.sa4(x4)
        x5, x5idx = sa4.max(dim=-1, keepdim=False)
        x5idx = x5idx[0][0].cpu().numpy()
        x4 = sa4 * x4
        # x5,x5index = x4.max(dim=-1, keepdim=False)
        # x5index = x5index[0].cpu().numpy()
        # x5 = x4.max(dim=1, keepdim=False)[0]
        # x6, x6index = x5.max(dim=-1, keepdim=False)
        # x6index = x6index[0].cpu().numpy()
        x4 = x4.mean(dim=-1, keepdim=False)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)# (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)

        max_values, indices = torch.max(x, dim=2)
        x1 = max_values
        #x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,-1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,-1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x, indices, [x11index, x12index, x13index],idx,x5idx


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all):
        super(DGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.transform_net = Transform_Net(args)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.ca1 = ChannelAttention(in_planes=64)
        self.ca2 = ChannelAttention(in_planes=64)
        self.ca3 = ChannelAttention(in_planes=64)


        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()

        self.bn1_ = nn.Sequential(nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))
        self.bn2_ = nn.Sequential(nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))
        self.bn3_ = nn.Sequential(nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))


        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x, l):
        xyz = x
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)  # (batch_size, 3, 3)
        x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x11 = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x12 = x11[:, :, :, 0:30]
        x13 = x11[:, :, :, 0:20]
        x14 = x11[:, :, :, 0:10]
        x11 = x11.max(dim=-1, keepdim=False)[0]
        x12 = x12.max(dim=-1, keepdim=False)[0]
        x13 = x13.max(dim=-1, keepdim=False)[0]
        x14 = x14.max(dim=-1, keepdim=False)[0]


        x1 = torch.stack((x11, x12, x13, x14), dim=3)
        x1 = self.bn1_(x1)
        ca1 = self.ca1(x1)
        x1 = ca1 * x1
        # x1 = self.sf1(x1)
        sa1 = self.sa1(x1)
        x1 = sa1 * x1
        x1 = x1.mean(dim=-1, keepdim=False)
        x1 = (x1 + x11 + x12 + x13 + x14) / 5

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x11 = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x12 = x11[:, :, :, 0:30]
        x13 = x11[:, :, :, 0:20]
        x14 = x11[:, :, :, 0:10]
        x11 = x11.max(dim=-1, keepdim=False)[0]
        x12 = x12.max(dim=-1, keepdim=False)[0]
        x13 = x13.max(dim=-1, keepdim=False)[0]
        x14 = x14.max(dim=-1, keepdim=False)[0]
        x2 = torch.stack((x11, x12, x13, x14), dim=3)
        x2 = self.bn2_(x2)
        ca2 = self.ca2(x2)
        x2 = ca2 * x2
        # x1 = self.sf1(x1)
        sa2 = self.sa2(x2)
        x2 = sa2 * x2
        x2 = x2.mean(dim=-1, keepdim=False)
        x2 = (x2 + x11 + x12 + x13 + x14) / 5

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x11 = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x12 = x11[:, :, :, 0:30]
        x13 = x11[:, :, :, 0:20]
        x14 = x11[:, :, :, 0:10]
        x11 = x11.max(dim=-1, keepdim=False)[0]
        x12 = x12.max(dim=-1, keepdim=False)[0]
        x13 = x13.max(dim=-1, keepdim=False)[0]
        x14 = x14.max(dim=-1, keepdim=False)[0]
        x3 = torch.stack((x11, x12, x13, x14), dim=3)
        x3 = self.bn3_(x3)
        ca3 = self.ca3(x3)
        x3 = ca3 * x3
        # x1 = self.sf1(x1)
        sa3 = self.sa3(x3)
        x3 = sa3 * x3
        x3 = x3.mean(dim=-1, keepdim=False)
        x3 = (x3 + x11 + x12 + x13 + x14) / 5
        # list = [l, xyz[0]]
        # np_array_list = [tensor.detach().cpu().numpy() for tensor in list]
        # np_array = np.array(np_array_list)
        # np.save('segmentation.npy', np_array)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        return x


class DGCNN_semseg(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature_cross


class STNkd(nn.Module):
    def __init__(self, args, d=64):
        super(STNkd, self).__init__()
        self.args = args
        
        self.conv1 = VNLinearLeakyReLU(d, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128//3, 1024//3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU(1024//3, 512//3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512//3, 256//3, dim=3, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(1024//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.fc3 = VNLinear(256//3, d)
        self.d = d

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x


class VN_PointNet_AM(nn.Module):
    def __init__(self, args, global_feat=True, feature_transform=False):
        super(VN_PointNet_AM, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        self.feat_dim = args.feat_dim
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        
        self.conv3 = VNLinear(128//3, self.feat_dim)
        self.bn3 = VNBatchNorm(self.feat_dim, dim=4)
        
        #self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        if self.feature_transform:
            self.fstn = STNkd(args, d=64//3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_2_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.complex_lin_1 = VN_PointNet_Affine_Module(self.feat_dim, self.feat_dim, args=self.args)

        self.vnn_lin_3 = VNSimpleLinear(2 *self.feat_dim, self.feat_dim)
        self.vnn_lin_4 = VNSimpleLinear(2 *self.feat_dim, self.feat_dim)
        self.vnn_lin_4_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.complex_lin_2 = VN_PointNet_Affine_Module(self.feat_dim, self.feat_dim, args=self.args)


    def forward(self, x):
        device = torch.device("cuda")
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat)
        x = self.pool(x)
        
        x = self.conv1(x)  # B,21,3,N
        
        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1,1,1,N)
            x = torch.cat((x, x_global), 1)
        
        #pointfeat = x
        x = self.conv2(x)     # B,128//3,3,N
        x = self.bn3(self.conv3(x))  # B,dim,3,N
        
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)  # B,dim *2 ,3,N

        #x, trans = self.std_feature(x)
        #x = x.view(B, -1, N)
        
        x = torch.max(x, -1, keepdim=False)[0]  # b,2*dim,3

        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3)  # [batch, dim * 2, 3, 1]
        # self.vnn_lin_1 就是VNLinearLeakyReLU 少了BR ，只有L
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3]

        # 使用linear从先前的特征中学习6D参数。
        j = self.vnn_lin_2(x)  # [batch, dim, 3, 1]
        j_ = self.vnn_lin_2_(x)  # [batch, dim, 3, 1]
        j = j.permute(0, 3, 1, 2)
        j_ = j_.permute(0, 3, 1, 2)
        j_6d = torch.stack([j, j_], dim=-1)  # [batch, dim, 3, 2]
        y = self.complex_lin_1(v, j_6d, device)  # B x dim * 2 x 3
        comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)

        # second
        x = self.vnn_lin_3(comb).permute(0, 3, 1, 2)

        # 使用linear从先前的特征中学习6D参数。
        j_2 = self.vnn_lin_4(comb)
        j_2_ = self.vnn_lin_4_(comb)
        j_2 = j_2.permute(0, 3, 1, 2)
        j_2_ = j_2_.permute(0, 3, 1, 2)
        j_2_6d = torch.stack([j_2, j_2_], dim=-1)  # [batch,N, dim , 3, 2]
        y_2 = self.complex_lin_2(x, j_2_6d, device)
        comb = torch.cat((x.permute(0, 2, 3, 1), y_2), dim=1)


        x = comb.squeeze(3)

        trans_feat = None
        '''if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat'''

        return x, trans_feat
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature


class get_model(nn.Module):
    def __init__(self, args, num_class=40, normal_channel=False):
        super(get_model, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 128//3)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 256//3)

        self.vnn_lin_1 = VNSimpleLinear(64//3 + 64//3 + 128//3 + 256//3, 512 // 3)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(64//3 + 64//3 + 128//3 + 256//3, 512 // 3)
        self.complex_lin_1 = ComplexLinearAndLeakyReLU(512 // 3, 512 // 3)

        self.vnn_lin_3 = VNSimpleLinear(2*(512 // 3), 512 // 3)
        self.vnn_lin_4 = VNSimpleLinear(2*(512 // 3), 512 // 3)
        self.complex_lin_2 = ComplexLinearAndLeakyReLU(512 // 3, 512 // 3)

        self.conv5 = VNLinearLeakyReLU(2*(512 // 3), 1024//3, dim=4, share_nonlinearity=True)
        
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((1024//3)*12, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)
        
        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    def forward(self, x):

        batch_size = x.size(0)
        x = x.unsqueeze(1)

        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = get_graph_feature(x3, k=self.n_knn)
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x1234 = torch.cat((x1, x2, x3, x4), dim=1)  # b,169,3,1024

        # >> 第1个Ori模块位置  (比较建议第一个)
        # >>>>>>>>> orientation-Aware Mechanism
        # first
        # self.vnn_lin_1 就是VNLinear
        x = self.vnn_lin_1(x1234).permute(0, 3, 1, 2)  # (b, 1024, dim//3, 3)
        j = self.vnn_lin_2(x1234).permute(0, 3, 1, 2)  # (b, 1024, dim//3, 3)
        # f_complexlinear: y = R(j) @ Z(A,B,C) @ R(j).T @ x
        y = self.complex_lin_1(x, j)  # B x dim/3 x 3 x 1024
        comb = torch.cat((x.permute(0, 2, 3, 1), y), dim=1)   # (b, dim//3 *2, 3, 1024)

        # second
        x = self.vnn_lin_3(comb).permute(0, 3, 1, 2)
        j2 = self.vnn_lin_4(comb).permute(0, 3, 1, 2)
        y2 = self.complex_lin_2(x, j2)
        comb = torch.cat((x.permute(0, 2, 3, 1), y2), dim=1)

        x = self.conv5(comb)  # b,341,3,1024

        # >> 第2个Ori模块的可选位置
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)

        # inv feature
        # 为什么不pooling之后再弄不变特征，而是要带着1024个点一起做？
        x, trans = self.std_feature(x)

        # globa pooling
        x = x.view(batch_size, -1, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # head
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        trans_feat = None
        return x, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, smoothing=True):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        target = target.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')
            
        return loss

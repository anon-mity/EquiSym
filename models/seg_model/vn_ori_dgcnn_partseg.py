import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature


class get_model(nn.Module):
    def __init__(self, args, num_part=50, normal_channel=False):
        super(get_model, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv4 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv5 = VNLinearLeakyReLU(64//3*2, 64//3)

        self.vnn_lin_1 = VNSimpleLinear(64 // 3 + 64 // 3 + 64//3, 512 // 3)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(64 // 3 + 64 // 3 + 64//3, 512 // 3)
        self.complex_lin_1 = ComplexLinearAndLeakyReLU(512 // 3, 512 // 3)

        self.vnn_lin_3 = VNSimpleLinear(2 * (512 // 3), 512 // 3)
        self.vnn_lin_4 = VNSimpleLinear(2 * (512 // 3), 512 // 3)
        self.complex_lin_2 = ComplexLinearAndLeakyReLU(512 // 3, 512 // 3)
        
        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
        
        self.conv6 = VNLinearLeakyReLU(2 * (512 // 3), 1024//3, dim=4, share_nonlinearity=True)
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        self.conv8 = nn.Sequential(nn.Conv1d(2299, 256, kernel_size=1, bias=False),
                               self.bn8,
                               nn.LeakyReLU(negative_slope=0.2))
        
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, num_part, kernel_size=1, bias=False)
        

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)
        
        x = x.unsqueeze(1)
        
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)

        x123 = torch.cat((x1, x2, x3), dim=1)  # B,c,3,2048

        # >>>>>>>>> orientation-Aware Mechanism
        # first
        # self.vnn_lin_1 就是VNLinearLeakyReLU 少了BR ，只有L
        x = self.vnn_lin_1(x123).permute(0, 3, 1, 2)  # (b, 1024, dim/3, 3)
        # self.vnn_lin_2 就是VNLinearLeakyReLU 少了BR ，只有L
        j = self.vnn_lin_2(x123).permute(0, 3, 1, 2)  # (b, 1024, dim/3, 3)
        # f_complexlinear: y = R(j) @ Z(A,B,C) @ R(j).T @ x
        y = self.complex_lin_1(x, j)  # B x dim/3 x 3 x 1024
        comb = torch.cat((x.permute(0, 2, 3, 1), y), dim=1)

        # second
        x = self.vnn_lin_3(comb).permute(0, 3, 1, 2)
        j_2 = self.vnn_lin_4(comb).permute(0, 3, 1, 2)
        y_2 = self.complex_lin_2(x, j_2)
        comb = torch.cat((x.permute(0, 2, 3, 1), y_2), dim=1)

        x = self.conv6(comb)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        # 得到VN-DGCNN之后的特征
        x = torch.cat((x, x_mean), 1)

        x, z0 = self.std_feature(x)

        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)
        x = x.view(batch_size, -1, num_points)
        x = x.max(dim=-1, keepdim=True)[0]

        l = l.view(batch_size, -1, 1)
        l = self.conv7(l)

        x = torch.cat((x, l), dim=1)
        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x123), dim=1)

        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)
        
        trans_feat = None
        return x.transpose(1, 2), trans_feat


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
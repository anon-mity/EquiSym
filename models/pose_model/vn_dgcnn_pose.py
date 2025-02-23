import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature

class VN_DGCNN(nn.Module):
    def __init__(self,cfg):
        super(VN_DGCNN, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.VnInv = VNStdFeature(self.feat_dim * 2, dim=3, normalize_frame=False)
        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64//3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):

        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)

        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123)  # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)

        x = self.pool4(x)  # [batch, 2*dim, 3]

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]


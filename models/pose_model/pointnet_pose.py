import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.pointnet import PointNetEncoder, feature_transform_reguliarzer



class PointNet(nn.Module):
    def __init__(self, args):
        super(PointNet, self).__init__()
        channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        inv_x = None
        return x, inv_x   # b,feat_dim

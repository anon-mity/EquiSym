import torch
import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.feat_dim = 2 * cfg.feat_dim
        #if cfg.model == 'pointnet':
        #    self.feat_dim = cfg.feat_dim
        if ((cfg.model =='vn_dgcnn' or cfg.model =='vn_pointnet' or cfg.model == 'vn_pointnet_am' or cfg.model == 'vn_ori_dgcnn' or
            cfg.model == 'vn_ori_globa' or cfg.model == 'vn_ori_globa6d' or cfg.model == 'vn_ori_globa9d' or cfg.model == 'vn_localori_globa6d'
            or cfg.model == 'vn_transformer_am') or cfg.model == 'vn_transformer_amx1' or cfg.model == 'vn_transformer_amx3' or
                cfg.model == 'abla_vntrans_wo_rotation' or cfg.model == 'abla_vntrans_wo_complex' or cfg.model == 'abla_vntrans_wo_aggregation'or
            cfg.model == 'abla_vntrans_eulur' or cfg.model == 'abla_vntrans_quat' or cfg.model == 'abla_vntrans_axangle') :
            self.feat_dim = cfg.feat_dim * 2 * 3

        elif cfg.model == 'vn_transformer' or cfg.model =='vn_ori_globa6d_res' or cfg.model =='vn_transformer_amx3_res':
            self.feat_dim = cfg.feat_dim * 3
        if cfg.mode == 'registr':
            self.feat_dim = 2 * self.feat_dim
            
        self.out_dim = 6
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 6)
        )
    def forward(self,equ_feat):
        batch_size = equ_feat.shape[0]  # b, 2*dim, 3
        equ_feat = equ_feat.reshape(batch_size, -1)   # b, 2*dim * 3

        pred_r6d = self.fc_layer(equ_feat)
        pred_r33 = recover_R_from_6d(pred_r6d)

        return pred_r33


class VN_Regressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feat_dim = 2 * cfg.feat_dim
        if cfg.model == 'pointnet':
            self.feat_dim = cfg.feat_dim
        elif cfg.model == 'vn_dgcnn' or cfg.model == 'vn_ori_dgcnn' or cfg.model == 'vn_ori_globa':
            self.feat_dim = cfg.feat_dim * 2

        if cfg.mode == 'registr':
            self.feat_dim = self.feat_dim * 2

        self.out_dim = 2
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, equ_feat):
        batch_size = equ_feat.shape[0]  # b, 2*dim, 3
        equ_feat = equ_feat.permute(0,2,1)  # b, 3,2*dim

        pred_r6d = self.fc_layer(equ_feat)  # b, 3, 2
        pred_r33 = bgs(pred_r6d)

        return pred_r33



# 6D -> 3x3
def recover_R_from_6d(R_6d):
    # R_6d:  b, 6 >>>> B, 3, 2
    R_6d = R_6d.reshape(-1, 2, 3).permute(0, 2, 1)
    R = bgs(R_6d)
    # R is batch * 3 * 3
    return R

# b, 3, 2 的6D旋转向量（轴角），转化为b, 3, 3的旋转矩阵
# 这段代码主要基于第一个向量，取第二个向量的正交投影，再取外积得到第三个向量。堆叠而成3x3的旋转矩阵
def bgs(d6s):
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)
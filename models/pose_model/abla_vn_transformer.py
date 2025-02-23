import torch
import torch.nn as nn
#import vn_transformer.layers as vn
from models import vn_transformer_layers as vn
from models.vn_layers import *


class VN_Transformer_AMx1_Abla(nn.Module):
    # Described in Figure 2
    def __init__(self,
                 cfg,
                 dim,
                 num_heads=64,
                 in_features=1,
                 latent_size=64,
                 bias_eps=1e-6,
                ):
        super().__init__()

        self.ablamode = cfg.model.rsplit('_', 1)[-1]

        self.leaky = 0.0
        hidden_features = dim
        self.feat_dim = dim
        self.vn_mlp = nn.Sequential(
            vn.Linear(in_features, hidden_features, bias_eps),
            vn.BatchNorm(hidden_features),
            vn.LeakyReLU(hidden_features, self.leaky),
            vn.Linear(hidden_features, hidden_features, bias_eps),
            vn.BatchNorm(hidden_features),
            vn.LeakyReLU(hidden_features, self.leaky),
        )

        if latent_size is not None:
            self.query_proj = vn.MeanProject(latent_size, hidden_features, hidden_features)
        else:
            self.query_proj = nn.Identity()

        self.vn_transformer = vn.TransformerBlock(f_dim=hidden_features, num_heads=num_heads, bias_eps=bias_eps, leaky=self.leaky)


        self.vnn_lin_1 = VNSimpleLinear(self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.vnn_lin_2_ = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.complex_lin_1 = Affine_Module_Abla(self.feat_dim, self.feat_dim, args=None, ablamode=self.ablamode)

    def forward(self, x):
        device = torch.device("cuda")
        x = x.unsqueeze(1)  # B,3,N -> [B, 1, 3, N]

        '''
        x: tensor of shape [B, num_features, 3, num_points]
        return: tensor of shape [B, num_classes]
        '''
        x = self.vn_mlp(x)

        queries = self.query_proj(x)
        x = self.vn_transformer(x, queries)  # B, D, 3, 64(head_dim)

        # 转化为不变特征
        #x = vn.invariant(x, self.vn_mlp_inv(x))  # B, D, 3, 64(head_dim)

        x = vn.mean_pool(x)  # B, dim, 3 (全局特征)


        # >>> Affine Mechanism
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

        x = comb.squeeze(3)

        x_inv = None

        return x, x_inv


class VN_Transformer_AMx1_Abla_Eulur_Quat(nn.Module):
    # Described in Figure 2
    def __init__(self,
                 cfg,
                 dim,
                 num_heads=64,
                 in_features=1,
                 latent_size=64,
                 bias_eps=1e-6,
                ):
        super().__init__()

        self.ablamode = cfg.model.rsplit('_', 1)[-1]  # 'eulur', 'quat'

        self.leaky = 0.0
        hidden_features = dim
        self.feat_dim = dim
        self.vn_mlp = nn.Sequential(
            vn.Linear(in_features, hidden_features, bias_eps),
            vn.BatchNorm(hidden_features),
            vn.LeakyReLU(hidden_features, self.leaky),
            vn.Linear(hidden_features, hidden_features, bias_eps),
            vn.BatchNorm(hidden_features),
            vn.LeakyReLU(hidden_features, self.leaky),
        )

        if latent_size is not None:
            self.query_proj = vn.MeanProject(latent_size, hidden_features, hidden_features)
        else:
            self.query_proj = nn.Identity()

        self.vn_transformer = vn.TransformerBlock(f_dim=hidden_features, num_heads=num_heads, bias_eps=bias_eps, leaky=self.leaky)

        self.vnn_lin_1 = VNSimpleLinear(self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.vnn_lin_2_ = VNSimpleLinear(3, 1)
        self.complex_lin_1 = Affine_Module_Abla(self.feat_dim, self.feat_dim, args=None, ablamode=self.ablamode)

    def forward(self, x):
        device = torch.device("cuda")
        x = x.unsqueeze(1)  # B,3,N -> [B, 1, 3, N]

        '''
        x: tensor of shape [B, num_features, 3, num_points]
        return: tensor of shape [B, num_classes]
        '''
        x = self.vn_mlp(x)

        queries = self.query_proj(x)
        x = self.vn_transformer(x, queries)  # B, D, 3, 64(head_dim)

        # 转化为不变特征
        #x = vn.invariant(x, self.vn_mlp_inv(x))  # B, D, 3, 64(head_dim)

        x = vn.mean_pool(x)  # B, dim, 3 (全局特征)


        # >>> Affine Mechanism
        # first
        x = x.unsqueeze(3)  # [batch, dim * 2, 3, 1]
        # self.vnn_lin_1 就是VNLinearLeakyReLU 少了BR ，只有L
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, 1, dim * 2, 3]
        # 使用linear从先前的特征中学习欧拉角参数。
        #if self.ablamode == 'eulur':
        j = self.vnn_lin_2(x)  # [batch, dim, 3, 1]
        j = j.permute(0, 3, 1, 2)  # [batch, 1, dim, 3]

        if self.ablamode == 'quat' or self.ablamode == 'axangle':
            x_1 = x.permute(0,2,1,3)  # B,3,D,1
            w = self.vnn_lin_2_(x_1)  # B,1,D,1
            w = w.permute(0, 3, 2, 1) # B,1,D,1
            j = torch.cat((j, w),dim=-1)   # B,1,D,4

        y = self.complex_lin_1(v, j, device)  # B x dim * 2 x 3
        comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)

        x = comb.squeeze(3)

        x_inv = None

        return x, x_inv
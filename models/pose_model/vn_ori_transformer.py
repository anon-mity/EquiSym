import torch
import torch.nn as nn
#import vn_transformer.layers as vn
from models import vn_transformer_layers as vn
from models.vn_layers import *


class VN_Transformer_AM(nn.Module):
    # Described in Figure 2
    def __init__(self,
                 dim,
                 num_heads=64,
                 in_features=1,
                 latent_size=64,
                 bias_eps=1e-6,
                ):

        super().__init__()
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

        '''self.vn_mlp_inv = nn.Sequential(
            vn.Linear(hidden_features, 3, bias_eps),
            vn.LeakyReLU(3, self.leaky),
        )'''

        '''self.mlp = nn.Sequential(
            nn.Linear(hidden_features*3, hidden_features),
            nn.ReLU(True),
            nn.Linear(hidden_features, num_classes)
        )'''

        self.vnn_lin_1 = VNSimpleLinear(self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.vnn_lin_2_ = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.complex_lin_1 = Affine_Geometric_Module(self.feat_dim, self.feat_dim, args=None)

        self.vnn_lin_3 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_4 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_4_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.complex_lin_2 = Affine_Geometric_Module(self.feat_dim, self.feat_dim, args=None)

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

        x_inv = None

        return x, x_inv


class VN_Transformer_AMx1(nn.Module):
    # Described in Figure 2
    def __init__(self,
                 dim,
                 num_heads=64,
                 in_features=1,
                 latent_size=64,
                 bias_eps=1e-6,
                ):

        super().__init__()
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
        self.complex_lin_1 = Affine_Geometric_Module(self.feat_dim, self.feat_dim, args=None)


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


class VN_Transformer_AMx3(nn.Module):
    # Described in Figure 2
    def __init__(self,
                 dim,
                 num_heads=64,
                 in_features=1,
                 latent_size=64,
                 bias_eps=1e-6,
                ):

        super().__init__()
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
        self.complex_lin_1 = Affine_Geometric_Module(self.feat_dim, self.feat_dim, args=None)

        self.vnn_lin_3 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_4 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_4_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.complex_lin_2 = Affine_Geometric_Module(self.feat_dim, self.feat_dim, args=None)

        self.vnn_lin_5 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_6 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_6_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.complex_lin_3 = Affine_Geometric_Module(self.feat_dim, self.feat_dim, args=None)


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

        # third
        x = self.vnn_lin_5(comb).permute(0, 3, 1, 2)

        # 使用linear从先前的特征中学习6D参数。
        j_3 = self.vnn_lin_6(comb)
        j_3_ = self.vnn_lin_6_(comb)
        j_3 = j_3.permute(0, 3, 1, 2)
        j_3_ = j_3_.permute(0, 3, 1, 2)
        j_3_6d = torch.stack([j_3, j_3_], dim=-1)  # [batch,N, dim , 3, 2]
        y_3 = self.complex_lin_2(x, j_3_6d, device)
        comb = torch.cat((x.permute(0, 2, 3, 1), y_3), dim=1)

        x = comb.squeeze(3)

        x_inv = None

        return x, x_inv


class VN_Transformer_AMx3_Res(nn.Module):
    # Described in Figure 2
    def __init__(self,
                 dim,
                 num_heads=64,
                 in_features=1,
                 latent_size=64,
                 bias_eps=1e-6,
                ):

        super().__init__()
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
        self.complex_lin_1 = Affine_Geometric_Module(self.feat_dim, self.feat_dim, args=None)

        self.vnn_lin_3 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.vnn_lin_4 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.vnn_lin_4_ = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.complex_lin_2 = Affine_Geometric_Module(self.feat_dim, self.feat_dim, args=None)

        self.vnn_lin_5 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.vnn_lin_6 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.vnn_lin_6_ = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.complex_lin_3 = Affine_Geometric_Module(self.feat_dim, self.feat_dim, args=None)


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
        #comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)
        # 替换为残差连接
        comb = v.permute(0, 2, 3, 1) + y

        # second
        x = self.vnn_lin_3(comb).permute(0, 3, 1, 2)

        # 使用linear从先前的特征中学习6D参数。
        j_2 = self.vnn_lin_4(comb)
        j_2_ = self.vnn_lin_4_(comb)
        j_2 = j_2.permute(0, 3, 1, 2)
        j_2_ = j_2_.permute(0, 3, 1, 2)
        j_2_6d = torch.stack([j_2, j_2_], dim=-1)  # [batch,N, dim , 3, 2]
        y_2 = self.complex_lin_2(x, j_2_6d, device)
        #comb = torch.cat((x.permute(0, 2, 3, 1), y_2), dim=1)
        # 替换为残差连接
        comb = x.permute(0, 2, 3, 1) + y_2

        # third
        x = self.vnn_lin_5(comb).permute(0, 3, 1, 2)

        # 使用linear从先前的特征中学习6D参数。
        j_3 = self.vnn_lin_6(comb)
        j_3_ = self.vnn_lin_6_(comb)
        j_3 = j_3.permute(0, 3, 1, 2)
        j_3_ = j_3_.permute(0, 3, 1, 2)
        j_3_6d = torch.stack([j_3, j_3_], dim=-1)  # [batch,N, dim , 3, 2]
        y_3 = self.complex_lin_2(x, j_3_6d, device)
        #comb = torch.cat((x.permute(0, 2, 3, 1), y_3), dim=1)
        comb = x.permute(0, 2, 3, 1) + y_3

        x = comb.squeeze(3)
        x_inv = None

        return x, x_inv

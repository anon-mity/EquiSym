import torch
import torch.nn as nn
#import vn_transformer.layers as vn
from models import vn_transformer_layers as vn


class VN_Transformer(nn.Module):
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

    def forward(self, x):
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
        x_inv = None

        return x, x_inv
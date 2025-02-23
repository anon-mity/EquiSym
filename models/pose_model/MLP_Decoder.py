
import torch.nn as nn

class MLP_Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_pts = cfg.num_point
        self.fc_layers = nn.Sequential(
            nn.Linear(self.num_pts, self.num_pts * 2),
            nn.BatchNorm1d(self.num_pts * 2),
            nn.Tanh(),
            nn.Linear(self.num_pts * 2 , self.num_pts * 3),
        )
        
    def forward(self, Inv_feat):
        # inv_feat.shape (b,self.feat_dim ,feat_sim)
        batch_size = Inv_feat.shape[0]
        pts = self.fc_layers(Inv_feat)
        pts = pts.reshape(batch_size, self.num_pts, 3)
        return pts
        
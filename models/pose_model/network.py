import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from models.pose_model.dgcnn_pose import DGCNN
from models.pose_model.vn_dgcnn_pose import VN_DGCNN
from models.pose_model.vn_ori_dgcnn_pose import VN_Ori_DGCNN, VN_Ori_Globa, VN_Ori_Globa6D, VN_Ori_Globa9D, VN_LocalOri_Globa6D,VN_Ori_Globa6D_Res
from models.pose_model.pointnet_pose import PointNet
from models.pose_model.vn_pointnet import VN_PointNet
from models.pose_model.vn_ori_pointnet import VN_PointNet_AM

from models.pose_model.vn_transformer import VN_Transformer
from models.pose_model.vn_ori_transformer import VN_Transformer_AM,VN_Transformer_AMx1,VN_Transformer_AMx3,VN_Transformer_AMx3_Res
from models.pose_model.abla_vn_transformer import VN_Transformer_AMx1_Abla, VN_Transformer_AMx1_Abla_Eulur_Quat
from models.pose_model.MLP_Decoder import MLP_Decoder
from models.pose_model.Regressor import Regressor, VN_Regressor



# 旋转矩阵之间角度（弧度）的度量，也是测地线距离
def bgdR(Rgts, Rps):
    # input: R Matrix batch * 3 * 3
    Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
    Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1) #batch trace
    # necessary or it might lead to nans and the likes
    theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(theta)
    return angle

class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = self.init_encoder(cfg)
        self.encoder = torch.nn.DataParallel(self.encoder)

        self.regress = self.init_regress(cfg)
        self.regress = torch.nn.DataParallel(self.regress)

        if cfg.disentangle:
            self.decoder = self.init_decoder(cfg)
            self.decoder = torch.nn.DataParallel(self.decoder)

    def forward(self, pc_aug, pc_norm=None):
        # 等变和不变特征提取
        Equ_feat, Inv_feat = self.encoder(pc_aug)  # (b, 2*dim, 3)  # [batch, 2*dim, 1024] / None

        if pc_norm is not None:
            Equ_feat_norm, Inv_feat_norm = self.encoder(pc_norm)
            Equ_feat = torch.cat((Equ_feat, Equ_feat_norm), dim=1)  # (b, 2*dim, 3) -> (b, 4*dim, 3)

        # pose
        pred_r33 = self.regress(Equ_feat)

        # recons
        if self.cfg.disentangle:
            Inv_feat = torch.sum(Inv_feat, dim=1)  # b,dim,num_pts -> b ,num_pts
            recons_pc = self.decoder(Inv_feat)
            return pred_r33, recons_pc

        else:
            return pred_r33

    def training_step(self, pc_aug, pc_norm, label):
        # b,1024,3 -> b,3,1024
        src_pc = torch.permute(pc_aug,dims=(0, 2, 1))
        tgt_pc = torch.permute(pc_norm,dims=(0, 2, 1))
        if self.cfg.mode == 'registr':
            pred_r33 = self.forward(src_pc, tgt_pc)
            trans_pc = torch.einsum('bij,bjk->bik', pc_norm, pred_r33)
            sl_loss, _ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            angle_loss = torch.mean(bgdR(label, pred_r33))
            mse_loss = F.mse_loss(label, pred_r33)
            loss = angle_loss + sl_loss + mse_loss  # mse_loss

            loss_dict = {'loss': loss,
                         'sl_loss': sl_loss,
                         'angle_loss': angle_loss,
                         'mse_loss':mse_loss
                         }
            return loss_dict

        elif self.cfg.mode == 'pose' and not self.cfg.disentangle:
            pred_r33 = self.forward(src_pc)
            trans_pc = torch.einsum('bij,bjk->bik',pc_norm, pred_r33)
            sl_loss,_ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            #mse_loss = F.mse_loss(label, pred_r33)
            angle_loss = torch.mean(bgdR(label, pred_r33))
            loss = angle_loss  + sl_loss # mse_loss

            loss_dict = {'loss': loss,
                    'sl_loss': sl_loss,
                    #'mse_loss':mse_loss,
                    'angle_loss': angle_loss
             }
            return loss_dict

        else:
            pred_r33, recons_pc = self.forward(src_pc)
            trans_pc = torch.einsum('bij,bjk->bik', pc_norm, pred_r33)
            sl_loss, _ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            angle_loss = torch.mean(bgdR(label, pred_r33))

            recons_loss, _ = chamfer_distance(recons_pc, pc_norm, batch_reduction='mean', point_reduction='mean')

            loss = sl_loss + recons_loss + angle_loss

            loss_dict = {'loss': loss,
                         'sl_loss': sl_loss,
                         'angle_loss': angle_loss,
                         'recons_loss': recons_loss
                         }

            return loss_dict

    def test_step(self, pc_aug, pc_norm, label):
        # b,1024,3 -> b,3,1024
        src_pc = torch.permute(pc_aug, dims=(0, 2, 1))
        tgt_pc = torch.permute(pc_norm, dims=(0, 2, 1))

        if self.cfg.mode == 'registr':
            pred_r33 = self.forward(src_pc, tgt_pc)
            trans_pc = torch.einsum('bij,bjk->bik', pc_norm, pred_r33)
            sl_loss, _ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            angle_loss = torch.mean(bgdR(label, pred_r33))
            loss = angle_loss + sl_loss  # mse_loss

            loss_dict = {'loss': loss,
                         'sl_loss': sl_loss,
                         'angle_loss': angle_loss
                         }
            return loss_dict

        elif self.cfg.mode == 'pose' and not self.cfg.disentangle:
            pred_r33 = self.forward(src_pc.float())
            trans_pc = torch.einsum('bij,bjk->bik', pc_norm, pred_r33)
            sl_loss, _ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            angle_loss = torch.mean(bgdR(label, pred_r33))
            loss = sl_loss + angle_loss

            loss_dict = {'loss': loss,
                         'sl_loss': sl_loss,
                         #'mse_loss':mse_loss,
                         'angle_loss': angle_loss,
                         }
            return loss_dict

        else:
            pred_r33, recons_pc = self.forward(src_pc)
            trans_pc = torch.einsum('bij,bjk->bik', pc_norm, pred_r33)
            sl_loss, _ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            angle_loss = torch.mean(bgdR(label, pred_r33))

            recons_loss, _ = chamfer_distance(recons_pc, pc_norm, batch_reduction='mean', point_reduction='mean')

            loss = sl_loss + recons_loss + angle_loss

            loss_dict = {'loss': loss,
                         'sl_loss': sl_loss,
                         'angle_loss': angle_loss,
                         'recons_loss': recons_loss
                         }
            return loss_dict


    def init_encoder(self,cfg):
        encoder = DGCNN(cfg)
        if self.cfg.model == 'vn_dgcnn':
            encoder = VN_DGCNN(cfg)
        elif self.cfg.model == 'vn_ori_dgcnn':
            encoder = VN_Ori_DGCNN(cfg)
        elif self.cfg.model == 'vn_ori_globa':
            encoder = VN_Ori_Globa(cfg)
        elif self.cfg.model == 'vn_ori_globa6d':
            encoder = VN_Ori_Globa6D(cfg)
        elif self.cfg.model == 'vn_ori_globa6d_res':
            encoder = VN_Ori_Globa6D_Res(cfg)
        elif self.cfg.model == 'vn_ori_globa9d':
            encoder = VN_Ori_Globa9D(cfg)
        elif self.cfg.model == 'vn_localori_globa6d':
            encoder = VN_LocalOri_Globa6D(cfg)
        elif self.cfg.model == 'pointnet':
            encoder = PointNet(cfg)
        elif self.cfg.model == 'vn_pointnet':
            encoder = VN_PointNet(cfg)
        elif self.cfg.model == 'vn_pointnet_am':
            encoder = VN_PointNet_AM(cfg)
        elif self.cfg.model == 'vn_transformer':
            encoder = VN_Transformer(cfg.feat_dim)
        elif self.cfg.model == 'vn_transformer_am':
            encoder = VN_Transformer_AM(cfg.feat_dim)
        elif self.cfg.model == 'vn_transformer_amx1':
            encoder = VN_Transformer_AMx1(cfg.feat_dim)
        elif self.cfg.model == 'vn_transformer_amx3':
            encoder = VN_Transformer_AMx3(cfg.feat_dim)
        elif self.cfg.model == 'vn_transformer_amx3_res':
            encoder = VN_Transformer_AMx3_Res(cfg.feat_dim)
        elif self.cfg.model == 'abla_vntrans_wo_rotation' or self.cfg.model == 'abla_vntrans_wo_complex' or self.cfg.model == 'abla_vntrans_wo_aggregation':
            encoder = VN_Transformer_AMx1_Abla(cfg, cfg.feat_dim)
        elif self.cfg.model == 'abla_vntrans_eulur' or self.cfg.model == 'abla_vntrans_quat' or self.cfg.model == 'abla_vntrans_axangle':
            encoder = VN_Transformer_AMx1_Abla_Eulur_Quat(cfg, cfg.feat_dim)
        return encoder

    def init_decoder(self,cfg):
        decoder = MLP_Decoder(cfg)
        return decoder

    def init_regress(self,cfg):
        regressor = Regressor(cfg)
        if cfg.regress == 'vn':
            regressor = VN_Regressor(cfg)
        return regressor

    def test_metric(self, pc_aug, pc_norm, label):
        # b,1024,3 -> b,3,1024
        src_pc = torch.permute(pc_aug, dims=(0, 2, 1))
        tgt_pc = torch.permute(pc_norm, dims=(0, 2, 1))

        if self.cfg.mode == 'registr':
            pred_r33 = self.forward(src_pc, tgt_pc)
            trans_pc = torch.einsum('bij,bjk->bik', pc_norm, pred_r33)
            sl_loss, _ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            angle_loss = bgdR(label, pred_r33)
            jiaodu_loss = angle_loss * 180 / math.pi

            loss_dict = {'CD': sl_loss,
                         'RRE_hd': angle_loss,
                         'RRE_jd': jiaodu_loss
                         }
            return loss_dict

        elif self.cfg.mode == 'pose' and not self.cfg.disentangle:
            pred_r33 = self.forward(src_pc.float())
            trans_pc = torch.einsum('bij,bjk->bik', pc_norm, pred_r33)
            sl_loss, _ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            angle_loss = bgdR(label, pred_r33)
            jiaodu_loss = angle_loss * 180 / math.pi

            loss_dict = {'sl_loss': sl_loss,
                         'angle_loss': angle_loss,
                         'jiaodu': jiaodu_loss
                         }
            return loss_dict

    def test_robust(self, pc_norm):
        # b,1024,3 -> b,3,1024
        pc_norm = torch.permute(pc_norm, dims=(0, 2, 1))
        Equ_feat_norm, _ = self.encoder(pc_norm)  # B,D,3, _
        return Equ_feat_norm

    def latent_test(self, pc_aug, pc_norm=None):
        pc_aug = torch.permute(pc_aug, dims=(0, 2, 1))
        pc_norm = torch.permute(pc_norm, dims=(0, 2, 1))

        # 等变和不变特征提取
        Equ_feat, Inv_feat = self.encoder(pc_aug)  # (b, 2*dim, 3)  # [batch, 2*dim, 1024] / None
        if pc_norm is not None:
            Equ_feat_norm, Inv_feat_norm = self.encoder(pc_norm)
        else:
            Equ_feat_norm = None
        return Equ_feat, Equ_feat_norm
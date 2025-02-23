import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import euler_angles_to_matrix, axis_angle_to_matrix, quaternion_to_matrix


EPS = 1e-6

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        return x_out


class VNSimpleLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNSimpleLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


def get_on_vector(normalized_J):
    '''
    normalized_J: normalized direction vector [batch size, num points, embedding dimension, 3]
    '''

    # calculate normalized U vector that is orthogonal to J
    # Let J = [x, y, z]
    # Then U = [x, y, -(x^2 + y^2) / (z + eps)]

    # get [x, y]
    sub_vec_J = normalized_J[:, :, :, :2]  # b x c x e x 2
    sub_vec_J1 = sub_vec_J.unsqueeze(3)
    sub_vec_J2 = sub_vec_J.unsqueeze(4)
    # calculate (x^2 + y^2)
    U_z1 = torch.einsum("abcik,abckj->abcij",sub_vec_J1, sub_vec_J2)
    U_z = torch.squeeze(torch.squeeze(U_z1, 4), 3)  # b x c x e

    # calculate -(x^2 + y^2) / (z + eps)
    U_z = -U_z / (normalized_J[:, :, :, 2] + EPS)  # b x c x e

    # form [x, y, -(x^2 + y^2) / (z + eps)]
    U = torch.cat((sub_vec_J, U_z.unsqueeze(3)), dim=3)  # b x c x e x 3

    # normalize
    normalized_U = (U.permute(3, 0, 1, 2) / (torch.linalg.norm(U, dim=3) + EPS)).permute(1, 2, 3, 0)  # b x c x e x 3

    return normalized_U

def get_basis(J):
    '''
    J: direction vector [batch size (B), num points (C), embedding dimension (E), 3]
    '''
    J_mochang = torch.linalg.norm(J, dim=3) + EPS
    # normalize J vectors ,将 J 中的每个向量通过除以其模来归一化，得到单位向量
    normalized_J = (J.permute(3, 0, 1, 2) / J_mochang).permute(1, 2, 3, 0)  # b x c x e x 3

    normalized_U = get_on_vector(normalized_J)  # b x c x e x 3

    # calculate V vector that is orthogonal to J and U
    normalized_V = torch.cross(normalized_U, normalized_J, dim=3)  # b x c x e x 3

    # R = (U, V, J)
    R = torch.cat((normalized_U, normalized_V, normalized_J), dim=-1)  # b x c x e x 9
    B, C, E, _ = R.size()
    R = torch.reshape(R, (B, C, E, 3, 3))  # b x c x e x 3 x 3

    return R


def get_rtx(index, RT, X):
    '''
    R: rotation basis [b, c, e, 3, 3]
    X: point features of shape [B, C, E, 3]
    '''

    indexed_RT = RT[:, :, :, index, :]  # b x c x e x 3
    indexed_RT = torch.unsqueeze(indexed_RT, 3)
    X = torch.unsqueeze(X, 4)
    #rtx = torch.matmul(indexed_RT, X)  # b x c x e x 1 x 1
    rtx = torch.einsum("abcik,abckj->abcij",indexed_RT,X)
    rtx = torch.squeeze(torch.squeeze(rtx, 4), 3)  # b x c x e

    return rtx

def get_rrtx(index, R, RTX):
    '''
    R: rotation basis [b, c, e, 3, 3]
    RTX: [B, C, E]
    '''

    indexed_R = R[:, :, :, :, index].permute(3, 0, 1, 2)  # 3 x b x c x e
    rrtx = (indexed_R * RTX).permute(1, 2, 3, 0)  # b x c x e x 3

    return rrtx


# 论文中的旋转感知线性层
class ComplexLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexLinear, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., 1024, E, 3]
        J: directions of shape [..., 1024, E, 3] (same shape as X)
        '''

        # 1. 基于向量特征J计算一组正交特征
        R = get_basis(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.计算正交基特征矩阵，权重矩阵和投影矩阵的内积
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3

        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3

        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y



class ComplexLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(ComplexLinearAndLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = ComplexLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [..., C, E, 3]
        J: directions of shape [..., C, E, 3] (same shape as X)
        '''
        x = self.linear(X, J).permute(0, 2, 3, 1)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out



# 6D -> 3x3
def bgs_extended(d6s):
    """
    将6D旋转向量转换为3x3旋转矩阵。

    Args:
        d6s (torch.Tensor): 输入张量，形状为 (b, 1, dim, 3, 2)

    Returns:
        torch.Tensor: 旋转矩阵，形状为 (b, 1, dim, 3, 3)
    """
    # 确保输入形状为 (b, 1, dim, 3, 2)
    assert d6s.dim() == 5 and d6s.shape[-2] == 3 and d6s.shape[-1] == 2, \
        "输入张量形状应为 (b, 1, dim, 3, 2)"

    # 第一步：归一化第一个向量 b1
    # d6s[..., :, 0] 的形状为 (b, 1, dim, 3)
    b1 = F.normalize(d6s[..., :, 0], p=2, dim=-1)  # 形状: (b, 1, dim, 3)

    # 第二步：提取第二个向量 a2
    a2 = d6s[..., :, 1]  # 形状: (b, 1, dim, 3)

    # 第三步：计算 a2 在 b1 上的投影
    # 计算点积 <a2, b1>，形状为 (b, 1, dim, 1)
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)

    # 投影向量：<a2, b1> * b1，形状为 (b, 1, dim, 3)
    proj = dot * b1

    # 第四步：计算正交化后的 b2 向量
    b2_unorm = a2 - proj  # 去除在 b1 上的分量，形状: (b, 1, dim, 3)

    # 归一化得到 b2，形状: (b, 1, dim, 3)
    b2 = F.normalize(b2_unorm, p=2, dim=-1)

    # 第五步：计算第三个基向量 b3，通过 b1 和 b2 的外积
    b3 = torch.cross(b1, b2, dim=-1)  # 形状: (b, 1, dim, 3)

    # 第六步：堆叠 b1, b2, b3 生成旋转矩阵
    # 堆叠后形状为 (b, 1, dim, 3, 3)
    rot = torch.stack([b1, b2, b3], dim=-1)

    return rot


# Eulur -> 3x3
def eulur2matrix(J):
    """
        将形状为 (B, N, D, 3) 的欧拉角张量转换为旋转矩阵，输出形状为 (B, N, D, 3, 3)。

        参数：
        euler_angles (torch.Tensor): 形状为 (B, N, D, 3) 的欧拉角张量，单位为弧度。
        convention (str): 欧拉角的旋转顺序，例如 'ZYX'。

        返回：
        torch.Tensor: 形状为 (B, N, D, 3, 3) 的旋转矩阵张量。
        """
    # 确保输入是浮点型
    euler_angles = J.float()

    # 使用 PyTorch3D 的函数进行转换
    R = euler_angles_to_matrix(euler_angles, convention='ZYX')  # Shape: (B, N, D, 3, 3)

    '''# 分离欧拉角
    alpha = J[..., 0]  # 绕 Z 轴
    beta = J[..., 1]  # 绕 Y 轴
    gamma = J[..., 2]  # 绕 X 轴

    # 计算三角函数
    c_alpha = torch.cos(alpha)
    s_alpha = torch.sin(alpha)
    c_beta = torch.cos(beta)
    s_beta = torch.sin(beta)
    c_gamma = torch.cos(gamma)
    s_gamma = torch.sin(gamma)

    # 构建旋转矩阵的各个元素
    R = torch.zeros(B, N, D, 3, 3, device=J.device, dtype=J.dtype)
    R[..., 0, 0] = c_alpha * c_beta
    R[..., 0, 1] = c_alpha * s_beta * s_gamma - s_alpha * c_gamma
    R[..., 0, 2] = c_alpha * s_beta * c_gamma + s_alpha * s_gamma

    R[..., 1, 0] = s_alpha * c_beta
    R[..., 1, 1] = s_alpha * s_beta * s_gamma + c_alpha * c_gamma
    R[..., 1, 2] = s_alpha * s_beta * c_gamma - c_alpha * s_gamma

    R[..., 2, 0] = -s_beta
    R[..., 2, 1] = c_beta * s_gamma
    R[..., 2, 2] = c_beta * c_gamma'''

    return R

# Quat -> 3x3
def quat2matrix(J):
    B = J.shape[0]
    N = J.shape[1]
    D = J.shape[2]

    # 归一化四元数
    J = F.normalize(J, p=2, dim=-1)

    # 分离四元数的虚部 (x, y, z) 和实部 (w)
    x = J[..., 0]
    y = J[..., 1]
    z = J[..., 2]
    w = J[..., 3]

    # 转换为旋转矩阵
    quaternions_p3d = torch.stack([w, x, y, z], dim=-1)
    quaternions_flat = quaternions_p3d.view(-1, 4)
    rot_matrices_flat = quaternion_to_matrix(quaternions_flat)

    # 重新调整为 (B, N, D, 3, 3)
    R = rot_matrices_flat.view(B, N, D, 3, 3)

    '''# 计算旋转矩阵的元素
    # 第一行
    r00 = 1 - 2 * (y ** 2 + z ** 2)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)

    # 第二行
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x ** 2 + z ** 2)
    r12 = 2 * (y * z - x * w)

    # 第三行
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x ** 2 + y ** 2)

    # 堆叠旋转矩阵的各个元素
    # 形状为 (B, N, D, 3, 3)
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)'''

    return R


# Axangle -> 3x3
def Axangle2matrix(J):
    """
       将形状为 (B, N, D, 4) 的轴角张量转换为旋转矩阵，输出形状为 (B, N, D, 3, 3)。

       参数：
       J (torch.Tensor): 形状为 (B, N, D, 4) 的轴角张量，最后一维为 (u_x, u_y, u_z, theta)。

       返回：
       torch.Tensor: 形状为 (B, N, D, 3, 3) 的旋转矩阵张量。
       """
    B = J.shape[0]
    N = J.shape[1]
    D = J.shape[2]

    # 分离旋转轴 (u_x, u_y, u_z) 和旋转角度 theta
    U = J[..., :3]   #(B, N, D, 3)
    theta = J[..., 3].unsqueeze(-1)  # (B, N, D, 1)

    # 归一化旋转轴
    U = F.normalize(U, p=2, dim=-1)  # 形状: (b, 1, dim, 3)

    # 计算旋转向量: (u_x * theta, u_y * theta, u_z * theta)
    rot_vec = U * theta  # Shape: (B, N, D, 3)

    # 转换为旋转矩阵
    rot_vec_flat = rot_vec.view(-1, 3)  # (B*N*D, 3)
    rot_mat_flat = axis_angle_to_matrix(rot_vec_flat)  # Shape: (B*N*D, 3, 3)

    # 重新调整为 (B, N, D, 3, 3)
    R = rot_mat_flat.view(B,N,D, 3, 3)  # Shape: (B, N, D, 3, 3)

    return R

# 仿射线性层
class Affine_Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3


        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y

# 仿射线性层：标量复数
class Affine_Linear_hbfushu1(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(Affine_Linear_hbfushu1, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.D = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

        self.r = nn.Parameter(torch.randn(1, args.feat_dim))  # B, 1, D
        self.theta = nn.Parameter(torch.randn(1, args.feat_dim))
        self.relu = nn.ReLU()

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造标量复数的线性组合
        b = RT0X.shape[0]
        d = RT0X.shape[2]
        cos = torch.cos(torch.tanh(self.theta) * 3.14) # 1,e
        sin = torch.sin(torch.tanh(self.theta) * 3.14) # 1,e
        rcos = self.relu(self.r) * cos   # 1,e
        rsin = self.relu(self.r) * sin   # 1,e
        rcos = rcos.expand(b,1,d)
        rsin = rsin.expand(b,1,d)

        a_term = RT0X * rcos - RT1X * rsin   # b x c x e
        a_term = a_term.unsqueeze(3)

        b_term = RT0X * rsin + RT1X * rcos   # b x c x e
        b_term = b_term.unsqueeze(3)

        c_term = RT2X   # b x c x e
        c_term = c_term.unsqueeze(3)  # b x c x e x 1

        linear_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 1
        linear_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 1
        linear_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 1

        pc_term = torch.cat((linear_a_term, linear_b_term, linear_c_term), dim=-1)
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.D, pc_term)  # b x c x e' x 3

        Y = summed_c_term  # b x c x e' x 3

        return Y

# 仿射线性层：向量复数
class Affine_Linear_hbfushu2(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(Affine_Linear_hbfushu2, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.D = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

        self.r = nn.Parameter(torch.randn(1, args.feat_dim, 3))  # 1, D, 3
        #self.theta = nn.Parameter(torch.randn(1, args.feat_dim, 3))
        self.relu = nn.ReLU()

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造标量复数的线性组合
        b = RT0X.shape[0]
        d = RT0X.shape[2]
        #cos = torch.cos(torch.tanh(self.theta) * 3.14)  # 1,e, 3
        #sin = torch.sin(torch.tanh(self.theta) * 3.14)  # 1,e, 3
        #rcos = self.relu(self.r) * cos  # 1, e, 3
        #rsin = self.relu(self.r) * sin  # 1, e, 3
        rcos = self.relu(self.r)
        rsin = self.relu(self.r)
        rcos = rcos.expand(b, 1, d, 3)
        rsin = rsin.expand(b, 1, d, 3)

        a_term = get_rrtx(2, R, RT1X) * rcos - get_rrtx(1, R, RT2X) * rsin  # b x c x e x 3
        b_term = get_rrtx(1, R, RT1X) * rsin + get_rrtx(2, R, RT2X) * rcos # b x c x e x 3
        c_term = get_rrtx(0, R, RT0X)  # b x c x e x 3
        a_term = a_term + b_term

        linear_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        linear_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3

        Y = linear_a_term + linear_c_term  # b x c x e' x 3
        Y = torch.einsum('fe,bcei->bcfi', self.D, Y)  # b x c x e' x 3
        return Y



# 这一版本是使用其他的组合，使用1和2组合 然后把0单独拿出来，理论上应该和之前一样。
class Affine_Linear_X_YZ(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_X_YZ, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(1, R, RT1X) + get_rrtx(2, R, RT2X)  # b x c x e x 3
        b_term = get_rrtx(2, R, RT1X) - get_rrtx(1, R, RT2X)  # b x c x e x 3
        a_term = a_term + b_term
        c_term = get_rrtx(0, R, RT0X)  # b x c x e x 3

        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_c_term  # b x c x e' x 3

        Y = torch.einsum('fe,bcei->bcfi', self.B, Y)  # b x c x e' x 3

        return Y


# 这一版本是使用其他的组合，使用Z和X组合 然后把Y单独拿出来，理论上应该和之前一样。
class Affine_Linear_Y_ZX(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_Y_ZX, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(2, R, RT2X) + get_rrtx(0, R, RT0X)  # b x c x e x 3
        b_term = get_rrtx(0, R, RT2X) - get_rrtx(2, R, RT0X)  # b x c x e x 3
        c_term = get_rrtx(1, R, RT1X)  # b x c x e x 3

        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y

# 这一版本是对构造好的复数空间，使用单个A学习而不是AB两个
class Affine_Linear_onlyAC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_onlyAC, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        a_term = a_term + b_term
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3


        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        #summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_c_term  # b x c x e' x 3

        return Y

# 这一版本是对构造好的复数空间，使用单个A学习而不是AB两个
class Affine_Linear_onlyA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_onlyA, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3
        a_term = a_term + b_term + c_term


        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        #summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        #summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term   # b x c x e' x 3

        return Y

# 线性构造的方式改变，全用加法。
class Affine_Linear_add(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_add, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        # 注意下式的加法，本来是减法。
        b_term = get_rrtx(1, R, RT0X) + get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3


        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y


# 线性构造的方式改变，全用减法。
class Affine_Linear_subtr(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_subtr, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        # 注意下式的减法，本来是加法。
        a_term = get_rrtx(0, R, RT0X) - get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3


        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y


# 把旋转变成单位矩阵了
class Affine_Linear_Abla_worotation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_Abla_worotation, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        #R1 = bgs_extended(J)  # b x c x e x 3 x 3
        B = X.shape[0]
        N = X.shape[1]
        D = X.shape[2]
        identity_matrix = torch.eye(3, device=X.device).view(1, 1, 1, 3, 3)
        R = identity_matrix.expand(B, N, D, 3, 3)

        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3


        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y

# RT0X  RT1X RT2X 复制三份
class Affine_Linear_Abla_wocomplex(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_Abla_wocomplex, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        a_term = torch.cat((RT0X.unsqueeze(-1),RT1X.unsqueeze(-1),RT2X.unsqueeze(-1)),dim=-1)

        # 3.构造类复数的线性组合。
        #a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        #b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        #c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3


        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, a_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, a_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y

class Affine_Linear_Abla_woaggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_Abla_woaggregation, self).__init__()
        #self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        #self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        #self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3

        #summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        #summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        #summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = a_term + b_term + c_term  # b x c x e' x 3

        return Y


# 消融第一步旋转的4种方式
class Affine_Linear_Abla_Eulur(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_Abla_Eulur, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = eulur2matrix(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3

        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y

class Affine_Linear_Abla_Quat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_Abla_Quat, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 4] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = quat2matrix(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3

        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y

class Affine_Linear_Abla_Axangle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_Abla_Axangle, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 4] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = quat2matrix(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3

        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y
class Affine_Geometric_Module(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Affine_Geometric_Module, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 正常情况：使用XY轴构建复数空间，然后使用Z轴表示额外的自由度。并且XY轴分开使用线性层。
        #self.linear = Affine_Linear(in_channels, out_channels)

        # 这一版本是使用其他的组合，使用1和2组合 然后把0单独拿出来，理论上应该和之前一样。
        # 就是使用YZ 轴构建复数空间，然后使用X轴表示额外的自由度。
        self.linear = Affine_Linear_X_YZ(in_channels, out_channels)

        # 这一版本是使用Y_ZX的组合，其中使用ZX构建复数空间，然后使用Y表示单独的自由度。
        #self.linear = Affine_Linear_Y_ZX(in_channels, out_channels)


        # 线性构造的方式改变，全用加法。
        #self.linear = Affine_Linear_add(in_channels, out_channels)

        # 线性构造的方式改变，全用减法。
        #self.linear = Affine_Linear_subtr(in_channels, out_channels)

        # 标量复数构造
        #self.linear = Affine_Linear_hbfushu1(in_channels, out_channels, args)

        # 向量复数构造
        #self.linear = Affine_Linear_hbfushu2(in_channels, out_channels, args)

        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3,2] (same shape as X)
        '''
        x = self.linear(X, J).permute(0, 2, 3, 1)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out

class VN_PointNet_Affine_Module(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(VN_PointNet_Affine_Module, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 正常情况：使用XY轴构建复数空间，然后使用Z轴表示额外的自由度。并且XY轴分开使用线性层。
        self.linear = Affine_Linear(in_channels, out_channels)

        # 这一版本是对构造好的复数空间XY-Z，使用单个A学习XY而不是AB两个分开学习A和B 正常版本是上面的。
        #self.linear = Affine_Linear_onlyAC(in_channels, out_channels)

        # 复数空间为XY-Z , 但是全部只用一个A来学，而不是AB学习XY；C学习Z
        #self.linear = Affine_Linear_onlyA(in_channels, out_channels)

        # 这一版本是使用其他的组合，使用1和2组合 然后把0单独拿出来，理论上应该和之前一样。
        # 就是使用YZ 轴构建复数空间，然后使用X轴表示额外的自由度。
        self.linear = Affine_Linear_X_YZ(in_channels, out_channels)

        # 这一版本是使用Y_ZX的组合，其中使用ZX构建复数空间，然后使用Y表示单独的自由度。
        #self.linear = Affine_Linear_Y_ZX(in_channels, out_channels)

        # 线性构造的方式改变，全用加法。
        #self.linear = Affine_Linear_add(in_channels, out_channels)

        # 线性构造的方式改变，全用减法。
        #self.linear = Affine_Linear_subtr(in_channels, out_channels)

        # 标量复数构造
        #self.linear = Affine_Linear_hbfushu1(in_channels, out_channels, args)

        # 向量复数构造
        #self.linear = Affine_Linear_hbfushu2(in_channels, out_channels, args)

        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3,2] (same shape as X)
        '''
        x = self.linear(X, J).permute(0, 2, 3, 1)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class Affine_Module_Abla(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None, ablamode=''):
        super(Affine_Module_Abla, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope
        self.ablamode = ablamode

        # 正常情况：使用XY轴构建复数空间，然后使用Z轴表示额外的自由度。并且XY轴分开使用线性层。
        if self.ablamode == 'rotation':
            self.linear = Affine_Linear_Abla_worotation(in_channels, out_channels)
        elif self.ablamode == 'complex':
            self.linear = Affine_Linear_Abla_wocomplex(in_channels, out_channels)
        elif self.ablamode == 'aggregation':
            self.linear = Affine_Linear_Abla_woaggregation(in_channels, out_channels)

        # 旋转的方式，欧拉，四元数等
        elif self.ablamode == 'eulur':
            self.linear = Affine_Linear_Abla_Eulur(in_channels, out_channels)
        elif self.ablamode == 'quat':
            self.linear = Affine_Linear_Abla_Quat(in_channels, out_channels)
        elif self.ablamode == 'axangle':
            self.linear = Affine_Linear_Abla_Axangle(in_channels, out_channels)
        # 这一版本是使用其他的组合，使用1和2组合 然后把0单独拿出来，理论上应该和之前一样。
        # 就是使用YZ 轴构建复数空间，然后使用X轴表示额外的自由度。
        #self.linear = Affine_Linear_X_YZ(in_channels, out_channels)

        # 这一版本是使用Y_ZX的组合，其中使用ZX构建复数空间，然后使用Y表示单独的自由度。
        #self.linear = Affine_Linear_Y_ZX(in_channels, out_channels)


        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3,2] (same shape as X)
        if eulur J:B,N,D,3
        if quat / axangle  J:B,N,D,4
        '''
        x = self.linear(X, J).permute(0, 2, 3, 1)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out

# AR  input:B,D,3,N
class Local_Geometric(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(Local_Geometric, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.vnn_lin_1 = VNSimpleLinear(in_channels, out_channels)
        self.vnn_lin_2 = VNSimpleLinear(in_channels, out_channels)
        self.vnn_lin_2_= VNSimpleLinear(in_channels, out_channels)

        self.linear = Affine_Linear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, input, device):
        # input (32, 21, 3, 1024)

        v = self.vnn_lin_1(input).permute(0, 3, 1, 2)  # [batch, N, D, 3]
        # self.vnn_lin_2 就是VNLinearLeakyReLU 少了BR ，只有L
        j = self.vnn_lin_2(input)  # [batch, D , 3, N]
        j_ = self.vnn_lin_2_(input)  # [batch, D, 3, N]
        j = j.permute(0, 3, 1, 2)
        j_ = j_.permute(0, 3, 1, 2)
        j_6d = torch.stack([j, j_], dim=-1)  # [batch, N, D, 3, 2]

        x = self.linear(v, j_6d).permute(0, 2, 3, 1)
        # LeakyReLU

        x_out = self.leaky_relu(x)
        x_out = x_out.contiguous()
        #y = torch.cat((v.permute(0, 2, 3, 1), x_out), dim=1)
        return x_out


class Affine_Linear_9D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_9D, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J, S):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        S: [..., N, D, 3, 3 ] scaling
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        R = torch.einsum('bcefi,bceik->bcefk', R, S)
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3


        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y
class Affine_Geometric_Module_9D(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(Affine_Geometric_Module_9D, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = Affine_Linear_9D(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, S, device):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3,2] (same shape as X)
        S: point features of shape [..., N, D, 3,3]
        '''
        x = self.linear(X, J, S).permute(0, 2, 3, 1)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out

class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm', negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope
        
        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        
        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
from scipy.spatial.transform import Rotation as sciR
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance

def clipping_operator(X,v):
    N = X.shape[0]

    # Compute the dot product of each point with v
    dot_products = np.dot(X, v)

    # Find indices of the top N/2 points with highest dot(x, v) values
    N_half = N // 2
    sorted_indices = np.argsort(-dot_products)
    indices_to_remove = sorted_indices[:N_half]

    # Remove the selected points from X
    mask = np.ones(N, dtype=bool)
    mask[indices_to_remove] = False
    X_clipped = X[mask]

    return X_clipped
def upsample_point_cloud(points, target_num=1024):
    num_points = points.shape[0]
    num_new_points = target_num - num_points

    # 使用KNN找到每个点的最近邻
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)

    new_points = []
    for i in range(num_new_points):
        idx = i % num_points
        neighbor_idx = indices[idx, 1]
        # 计算中点
        new_point = (points[idx] + points[neighbor_idx]) / 2
        new_points.append(new_point)

    new_points = np.array(new_points)
    upsampled_points = np.vstack((points, new_points))
    return upsampled_points

def R_from_euler_np(angles):
    '''
    angles: [(b, )3]
    '''
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(angles[0]), -math.sin(angles[0]) ],
                    [0,         math.sin(angles[0]), math.cos(angles[0])  ]
                    ])
    R_y = np.array([[math.cos(angles[1]),    0,      math.sin(angles[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(angles[1]),   0,      math.cos(angles[1])  ]
                    ])

    R_z = np.array([[math.cos(angles[2]),    -math.sin(angles[2]),    0],
                    [math.sin(angles[2]),    math.cos(angles[2]),     0],
                    [0,                     0,                      1]
                    ])
    return np.dot(R_z, np.dot( R_y, R_x ))

def rotate_point_cloud(data, R = None, max_degree = None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        R:
          3x3 array, optional Rotation matrix used to rotate the input
        max_degree:
          float, optional maximum DEGREE to randomly generate rotation
        Return:
          Nx3 array, rotated point clouds
    """
    # rotated_data = np.zeros(data.shape, dtype=np.float32)
    if R is not None:
      rotation_angle = R
    elif max_degree is not None:
      rotation_angle = np.random.randint(0, max_degree, 3) * np.pi / 180.0
    else:
      rotation_angle = sciR.random().as_matrix() if R is None else R

    if isinstance(rotation_angle, list) or  rotation_angle.ndim == 1:
      rotation_matrix = R_from_euler_np(rotation_angle)
    else:
      assert rotation_angle.shape[0] >= 3 and rotation_angle.shape[1] >= 3
      rotation_matrix = rotation_angle[:3, :3]

    if data is None:
      return None, rotation_matrix
    rotated_data = np.dot(data, rotation_matrix)

    return rotated_data, rotation_matrix   # return [N, 3],

def R_z_only(angle):
    '''
    angle: single float, rotation angle around the Z-axis in radians
    '''
    return np.array([[math.cos(angle), -math.sin(angle), 0],
                     [math.sin(angle), math.cos(angle), 0],
                     [0, 0, 1]])

def rotate_point_cloud_z(data, R=None, max_degree=None):
    """ Randomly rotate the point clouds around the Z-axis to augment the dataset
        Input:
          Nx3 array, original point clouds
        R:
          3x3 array, optional Rotation matrix used to rotate the input
        max_degree:
          float, optional maximum DEGREE to randomly generate rotation
        Return:
          Nx3 array, rotated point clouds
    """
    if R is not None:
        rotation_angle = R
    elif max_degree is not None:
        rotation_angle = np.random.randint(0, max_degree) * np.pi / 180.0
    else:
        rotation_angle = np.random.random() * 2 * np.pi  # Random angle between 0 and 2pi

    # Generate Z-axis rotation matrix
    rotation_matrix = R_z_only(rotation_angle)

    if data is None:
        return None, rotation_matrix

    # Apply rotation
    rotated_data = np.dot(data, rotation_matrix)

    return rotated_data, rotation_matrix

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

# 归一化到一个以原点为中心，半径为1的单位球体范围内 [-1,1]
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ScanObjectNNH5(Dataset):
    def __init__(self, data_path, mode='train', category='', num_pts=1024):
        """
        Args:
            h5_file_path (str): Path to the h5 file containing the dataset.
            num_pts (int): Number of points to sample from each point cloud.
        """
        if mode == 'train':
            self.file_path = os.path.join(data_path, 'main_split/training_objectdataset_augmentedrot.h5')
        else:
            self.file_path = os.path.join(data_path, 'main_split/test_objectdataset_augmentedrot.h5')

        self.num_pts = num_pts


        # Load file
        with h5py.File(self.file_path, 'r') as f:
            self.data = f['data'][:]
            self.label = f['label'][:]
            self.label = f['label'][:]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        point_cloud = self.data[idx]

        # Randomly sample points
        if self.num_pts is not None and len(point_cloud) > self.num_pts:
            point_cloud = farthest_point_sample(point_cloud, self.num_pts)

        # normal
        point_cloud[:, 0:3] = pc_normalize(point_cloud[:, 0:3])

        # R augment
        pc = point_cloud
        pc_z, R_z = rotate_point_cloud_z(point_cloud[:, 0:3])  # z
        pc_so3, R_so3 = rotate_point_cloud(point_cloud[:, 0:3])  # so3

        # part_caiyang
        #v = np.random.randn(3)
        #v /= np.linalg.norm(v)
        #pc_part = clipping_operator(pc, v)
        #pc_part_1024 = upsample_point_cloud(pc_part)

        return {
            'pc_norm': torch.from_numpy(pc.astype(np.float32)),
            'pc_z': torch.from_numpy(pc_z.astype(np.float32)),
            'pc_so3': torch.from_numpy(pc_so3.astype(np.float32)),
            'target_z': torch.from_numpy(R_z.astype(np.float32)),
            'target_so3': torch.from_numpy(R_so3.astype(np.float32)),
        }

def vis(point_cloud_1_np,point_cloud_2_np,i,output_dir):

    fig = plt.figure(figsize=(20, 10))
    # 绘制第一个子图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(point_cloud_1_np[:, 0], point_cloud_1_np[:, 1], point_cloud_1_np[:, 2], c='r', marker='o')
    ax1.set_title('aug')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])

    # 绘制第二个子图
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(point_cloud_2_np[:, 0], point_cloud_2_np[:, 1], point_cloud_2_np[:, 2], c='g', marker='o')
    ax2.set_title('norm')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])

    # 保存图形到文件夹中
    output_file = os.path.join(output_dir, f'pc{i}.png')
    plt.savefig(output_file)
    plt.close()

def ppca_d(D):
    # 计算每个特征的均值
    mean_shape = np.mean(D, axis=0)  # 形状为 (3072,)
    # 中心化数据
    D_centered = D - mean_shape  # 形状仍为 (B, 3072)

    # 初始化PCA对象, 拟合PCA模型
    pca = PCA()
    pca.fit(D_centered)

if __name__ == "__main__":
    # dataset_path = "../../../DATA/ShapeNetAtlasNetH5_1024"
    dataset_path = "/home/hanbing/datasets/ScanObjectNN"
    n_points = 1024
    batch_size = 128
    data_set = ScanObjectNNH5(data_path=dataset_path, mode='test', category='plane', num_pts=1024)
    loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    for id, data in enumerate(loader, 0):
        pc_norm = data['pc_norm']
        #pc_norm = pc_norm.cuda()
        pc_z = data['pc_z']
        #pc_z = pc_z.cuda()
        pc_so3 = data['pc_so3']
        #pc_so3 = pc_so3.cuda()
        output_dir = '/home/hanbing/paper_code/vnn/visualizer/scanobject'
        for i in range(32):
            pc1 = pc_so3[i]
            pc2 = pc_norm[i]
            vis(pc1, pc2, i, output_dir)




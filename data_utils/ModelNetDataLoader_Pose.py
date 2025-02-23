import numpy as np
import warnings
import os
import math
from scipy.spatial.transform import Rotation as sciR
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings('ignore')


# 归一化到一个以原点为中心，半径为1的单位球体范围内 [-1,1]
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

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

class ModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=False, category_choice=None, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        if category_choice == 'lamp' :
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test_lamp.txt'))]
        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        fn = self.datapath[index]
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints,:]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.normal_channel:
            point_set = point_set[:, 0:3]

        pc_norm = point_set
        pc_so3, R_so3 = rotate_point_cloud(point_set[:, 0:3])  # so3

        return {'pc_norm': torch.from_numpy(pc_norm.astype(np.float32)),
                'pc_so3': torch.from_numpy(pc_so3.astype(np.float32)),
                'target_so3': torch.from_numpy(R_so3.astype(np.float32))}

    def __getitem__(self, index):
        return self._get_item(index)


class ModelNetDataLoader_Lamp(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=False, category_choice=None, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}

        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test_lamplatent.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        fn = self.datapath[index]
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints,:]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.normal_channel:
            point_set = point_set[:, 0:3]

        pc_norm = point_set
        pc_so3, R_so3 = rotate_point_cloud(point_set[:, 0:3])  # so3

        return {'pc_norm': torch.from_numpy(pc_norm.astype(np.float32)),
                'pc_so3': torch.from_numpy(pc_so3.astype(np.float32)),
                'target_so3': torch.from_numpy(R_so3.astype(np.float32))}

    def __getitem__(self, index):
        return self._get_item(index)

def vis(point_cloud_1_np,point_cloud_2_np,i,output_dir):

    fig = plt.figure(figsize=(10, 5))
    # 绘制第一个子图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(point_cloud_1_np[:, 0], point_cloud_1_np[:, 1], point_cloud_1_np[:, 2], c='r', marker='o')
    ax1.set_title('aug')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])

    # 绘制第二个子图
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(point_cloud_2_np[:, 0], point_cloud_2_np[:, 1], point_cloud_2_np[:, 2], c='g', marker='o')
    ax2.set_title('norm')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])

    # 保存图形到文件夹中
    output_file = os.path.join(output_dir, f'pc{i}.png')
    plt.savefig(output_file)

if __name__ == '__main__':
    import torch
    train_data = ModelNetDataLoader('/home/hanbing/datasets/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=False,
                                    category_choice='plane')
    train_DataLoader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

    test_data = ModelNetDataLoader('/home/hanbing/datasets/modelnet40_normal_resampled/', split='test', uniform=False,
                                    normal_channel=False, category_choice='plane')
    test_DataLoader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)
    for data in train_DataLoader:
        points = data['pc_so3']
        target = data['target_so3']
        point_norm = data['pc_norm']
        output_dir = '/home/hanbing/paper_code/vnn/visualizer/modelnet_plane'
        for i in range(16):
            pc1 = points[i]
            pc2 = point_norm[i]
            vis(pc1, pc2, i, output_dir)

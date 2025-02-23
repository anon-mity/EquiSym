import numpy as np
import glob
import os
from torch.utils.data import Dataset
import math
from scipy.spatial.transform import Rotation as sciR

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

# 使用该函数读3dmatch
class ThreedmatchDataset(Dataset):
  def __init__(self,
               data_path,
               split,
               OVERLAP_RATIO,
               point_limit=20000):

    self.root = data_path
    self.OVERLAP_RATIO = OVERLAP_RATIO
    self.files = []
    self.point_limit = point_limit
    self.aug_noise = 0.005

    data_files = f'{split}_3dmatch.txt'
    self.data_files = os.path.join(self.root, data_files)

    subset_names = open(self.data_files).read().split()   # 所有场景名
      # DATA_FILES = {
      #'train': './config/train_3dmatch.txt',
      #'val': './config/val_3dmatch.txt',
      #'test': './config/test_3dmatch.txt'
    self.root = os.path.join(self.root, 'threedmatch')
    for name in subset_names:
      fname = name + "*%.2f.txt" % self.OVERLAP_RATIO   #OVERLAP_RATIO = 0.3
      fnames_txt = glob.glob(self.root + "/" + fname)
      assert len(fnames_txt) > 0, f"Make sure that the path {self.root} has data {fname}"
      for fname_txt in fnames_txt:
        with open(fname_txt) as f:
          content = f.readlines()
        fnames = [x.strip().split() for x in content]   # fnames：list, 一个场景下的所有匹配对和重合率
        for fname in fnames:
          self.files.append([fname[0], fname[1]])


  def __len__(self):
      return len(self.files)

  def _load_point_cloud(self, points):
      # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
      if self.point_limit is not None and points.shape[0] > self.point_limit:
          indices = np.random.permutation(points.shape[0])[: self.point_limit]
          points = points[indices]
      return points


  def __getitem__(self, idx):
    file0 = os.path.join(self.root, self.files[idx][0])
    file1 = os.path.join(self.root, self.files[idx][1])
    data0 = np.load(file0)
    data1 = np.load(file1)

    # get point cloud
    points_norm0 = self._load_point_cloud(data0["pcd"])
    points_norm1 = self._load_point_cloud(data1["pcd"])

    # rotate
    points_aug0, R_0 = rotate_point_cloud(points_norm0)  # so3
    points_aug1, R_1 = rotate_point_cloud(points_norm1)  # so3
    R_rela = R_0.T @ R_1  # 将aug0 与 aug1对齐的旋转
    # points0_aglin = points_aug0 @ R_rela

    # noise
    points_aug0 += (np.random.rand(points_aug0.shape[0], 3) - 0.5) * self.aug_noise
    points_aug1 += (np.random.rand(points_aug1.shape[0], 3) - 0.5) * self.aug_noise

    return {'pc_aug0': torch.from_numpy(points_aug0.astype(np.float32)),
            'pc_aug1': torch.from_numpy(points_aug1.astype(np.float32)),
            #'points0_aglin': torch.from_numpy(points0_aglin.astype(np.float32)),
            "R_rela": torch.from_numpy(R_rela.astype(np.float32))}


if __name__ == '__main__':
    import torch

    train_data = ThreedmatchDataset('/home/hanbing/datasets/3Dmatch/',split='val', OVERLAP_RATIO=0.3)
    train_DataLoader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)

    test_data = ThreedmatchDataset('/home/hanbing/datasets/3Dmatch/', split='test', OVERLAP_RATIO=0.3)
    test_DataLoader = torch.utils.data.DataLoader(test_data, batch_size=2, shuffle=True)
    for data in train_DataLoader:
       #pc_norm0 = data['pc_norm0']
        #pc_norm1 = data['pc_norm1']
        pc_aug0 = data['pc_aug0']
        pc_aug1 = data['pc_aug1']
        points0_aglin = data["points0_aglin"]
        output_dir = '/home/hanbing/paper_code/vnn/visualizer/3dmatch'
        for i in range(2):
            #pc1 = pc_norm0[i].cpu().numpy()
            #pc2 = pc_norm1[i].cpu().numpy()
            pc3 = pc_aug0[i].cpu().numpy()
            pc4 = pc_aug1[i].cpu().numpy()
            pc5 = points0_aglin[i].cpu().numpy()

            #output_dir1 = os.path.join(output_dir, f'{i}pc1.txt')
            #output_dir2 = os.path.join(output_dir, f'{i}pc2.txt')
            output_dir3 = os.path.join(output_dir, f'{i}pc3.txt')
            output_dir4 = os.path.join(output_dir, f'{i}pc4.txt')
            output_dir5 = os.path.join(output_dir, f'{i}pc5.txt')

            #np.savetxt(output_dir1, pc1, fmt='%.6f', delimiter=' ')
            #np.savetxt(output_dir2, pc2, fmt='%.6f', delimiter=' ')
            np.savetxt(output_dir3, pc3, fmt='%.6f', delimiter=' ')
            np.savetxt(output_dir4, pc4, fmt='%.6f', delimiter=' ')
            np.savetxt(output_dir5, pc5, fmt='%.6f', delimiter=' ')
        break
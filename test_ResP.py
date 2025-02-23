import math
import csv
from data_utils.ShapeNetH5Dataloader import ShapeNetH5Loader
from data_utils.ModelNetDataLoader_Pose import ModelNetDataLoader
import argparse
import os
import torch
import torch.nn as nn
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
from tensorboardX import SummaryWriter
from models.pose_model.network import Network
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('')

    # mode (pose or registr)
    parser.add_argument('--mode', default='registr', choices=['pose', 'registr'])

    # model
    parser.add_argument('--model', default='vn_ori_globa6d', choices=['dgcnn', 'pointnet','vn_dgcnn',
                                                                      'vn_ori_dgcnn','vn_ori_globa','vn_ori_globa6d'])
    parser.add_argument('--feat_dim', default=512, type=int)
    parser.add_argument('--n_knn', default=20, type=int,
                        help='Number of nearest neighbors to use, not applicable to PointNet [default: 20]')
    parser.add_argument('--pooling', default='mean', type=str, help='VNN only: pooling method [default: mean]',
                        choices=['mean', 'max'])
    parser.add_argument('--regress', default='sn', choices=['sn', 'vn'])
    parser.add_argument('--disentangle', default=False)

    # Test
    parser.add_argument('--gpu', type=str, default="6,7")
    parser.add_argument('--checkpoint_dir', default='/home/hanbing/paper_code/vnn/log/registr/vn_ori_globa6d/so3_so3/2024-12-16_22-12')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size in training [default: 32]')

    # Setting
    parser.add_argument('--rot_test', type=str, default='so3', choices=['aligned', 'z', 'so3'])

    # Dataset
    parser.add_argument('--data_choice', default='shapenet', choices=['shapenet', 'modelnet'])
    parser.add_argument('--shapenet_path', default='/home/hanbing/datasets/ShapeNetAtlasNetH5_1024')
    parser.add_argument('--modelnet_path', default='/home/hanbing/datasets/modelnet40_normal_resampled')
    parser.add_argument('--category', type=str, default='plane')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()


def test(testDataLoader, pose_model, logger, args):
    def log_string(str):
        logger.info(str)
        print(str)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_num_batch = len(testDataLoader)  # batchsize * val_num_batch = val data num of one epoch

    mse = nn.MSELoss()
    mse_epoch = []
    '''Test'''
    logger.info('Start test...')
    with torch.no_grad():
        for test_batch_ind, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
            pc_norm = data['pc_norm']
            if args.rot_test == "so3":
                pc_aug = data['pc_so3']
                label = data['target_so3']
            else:
                pc_aug = data['pc_z']
                label = data['target_z']

            pc_norm = pc_norm.to(device).float()
            pc_aug = pc_aug.to(device).float()
            label = label.to(device).float()

            pose_model = pose_model.eval()
            feat_equ_norm = pose_model.module.test_robust(pc_norm)  # B, D, 3
            batchsize = feat_equ_norm.shape[0]

            ij_i = 0.0
            for i in range(batchsize):
                fxi = feat_equ_norm[i]
                # batchsize 定义一个形状x与多少个其他形状计算（假设为24）
                ij_j = 0.0
                for j in range(batchsize):
                    fxj = feat_equ_norm[j]
                    ij = mse(fxi, fxj)  # 单个对的mse
                    ij = ij.item()
                    ij_j += ij
                ij_j = ij_j / batchsize  # 一个实例的平均mse
                ij_i += ij_j
            ij_i = ij_i / batchsize  # 一个Batch实例的平均mse
            mse_batch = ij_i
            mse_epoch.append(mse_batch)

    Mean_mse_epoch = sum(mse_epoch)/len(mse_epoch)
    csv_file = os.path.join(args.checkpoint_dir,'ReSP.csv')
    headers = ['Category', 'ReSP1']
    row = [args.category, Mean_mse_epoch]

    # 检查文件是否存在
    file_exists = os.path.isfile(csv_file)

    # 打开文件并写入数据
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # 如果文件不存在，先写入表头
        if not file_exists:
            writer.writerow(headers)
        # 写入数据行
        writer.writerow(row)

def main(args, timestr):
    def log_string(str):
        logger.info(str)
        print(str)

    #categorys = ['bench', 'cabinet', 'car', 'cellphone', 'chair', 'couch', 'firearm',
    #             'lamp', 'monitor', 'plane', 'speaker','table', 'watercraft']

    if args.data_choice == 'shapenet':
        # 按照形变排序
        categorys = ['lamp', 'speaker', 'table', 'cabinet', 'chair', 'monitor', 'bench',
                     'watercraft', 'couch', 'plane', 'cellphone', 'firearm', 'car']
        categorys = ['lamp', 'speaker', 'table', 'cabinet', 'chair', 'monitor', 'bench',
                     'watercraft', 'couch', 'plane', 'cellphone', 'firearm', 'car']
        #categorys = [ 'lamp','table',  'chair',
        #              'plane',  'cellphone', 'car']
    else:
        categorys = ['all']  # 训练在modelnet的所有类别

    for i in range(len(categorys)):
        args.category = categorys[i]

        '''LOG'''
        logger = logging.getLogger("Model")
        logger.setLevel(logging.INFO)

        '''DATA LOADING'''
        if args.data_choice == 'shapenet':
            TEST_DATASET = ShapeNetH5Loader(data_path=args.shapenet_path, mode='val', category=args.category,
                                            num_pts=args.num_point)
            testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=4)
        else:
            TEST_DATASET = ModelNetDataLoader(root=args.modelnet_path, npoint=args.num_point, split='test',
                                              normal_channel=args.normal, category_choice=args.category)
            testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=4)

        '''MODEL LOADING'''
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        pose_model = Network(cfg=args).cuda()
        pose_model = nn.DataParallel(pose_model, device_ids=[0, 1])

        '''CheckPoints LOADING'''
        experiment_dir = args.checkpoint_dir
        checkpoint_dir = os.path.join(experiment_dir, args.category, 'checkpoints/best_model.pth')
        checkpoint = torch.load(checkpoint_dir)
        if isinstance(pose_model, torch.nn.DataParallel) or isinstance(pose_model, torch.nn.parallel.DistributedDataParallel):
            pose_model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            pose_model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')

        log_string(args)
        test(testDataLoader, pose_model, logger, args)

if __name__ == '__main__':
    args = parse_args()
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    main(args, timestr)

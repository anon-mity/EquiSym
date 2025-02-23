import math
import csv
from data_utils.ShapeNetH5Dataloader import ShapeNetH5Loader
from data_utils.ModelNetDataLoader_Pose import ModelNetDataLoader
from data_utils.ScanObjectNN_Pose import ScanObjectNNH5
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
    parser.add_argument('--model', default='vn_transformer_amx3_res',
                        choices=['dgcnn',
                                 'vn_dgcnn',
                                 'vn_ori_dgcnn','vn_ori_globa','vn_ori_globa6d','vn_ori_globa9d','vn_localori_globa6d',

                                 'pointnet',
                                 'vn_pointnet',
                                 'vn_pointnet_am',

                                 'vn_transformer',
                                 'vn_transformer_amx1',
                                 'vn_transformer_am',
                                 'vn_transformer_amx3',

                                 'abla_vntrans_wo_rotation',
                                 'abla_vntrans_wo_complex',
                                 'abla_vntrans_wo_aggregation',

                                 'abla_vntrans_eulur',  # 都写完了，明天debug下
                                 'abla_vntrans_quat',
                                 'abla_vntrans_axangle',

                                 'vn_ori_globa6d_res',
                                 'vn_transformer_amx3_res'
                                 ])

    parser.add_argument('--feat_dim', default=512, type=int)
    parser.add_argument('--n_knn', default=20, type=int,
                        help='Number of nearest neighbors to use, not applicable to PointNet [default: 20]')
    parser.add_argument('--pooling', default='mean', type=str, help='VNN only: pooling method [default: mean]',
                        choices=['mean', 'max'])
    parser.add_argument('--regress', default='sn', choices=['sn', 'vn'])
    parser.add_argument('--disentangle', default=False)

    # Test
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument('--checkpoint_dir', default='/home/hanbing/paper_code/vnn/log/registr/vn_transformer_amx3_res/so3_so3/2025-01-16_10-10')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size in training [default: 32]')

    # Setting
    parser.add_argument('--rot_test', type=str, default='so3', choices=['aligned', 'z', 'so3'])

    # Dataset
    parser.add_argument('--data_choice', default='shapenet',
                        choices=['shapenet', 'modelnet', 'scanobject', 'threedmatch'])
    parser.add_argument('--shapenet_path', default='/home/hanbing/datasets/ShapeNetAtlasNetH5_1024')
    parser.add_argument('--modelnet_path', default='/home/hanbing/datasets/modelnet40_normal_resampled')
    parser.add_argument('--scanobject_path', default='/home/hanbing/datasets/ScanObjectNN')
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

    # 定义旋转误差的阈值（根据具体任务设定）
    ROTATION_THRESHOLD5 = 5   # 示例值，需根据实际情况调整
    ROTATION_THRESHOLD10 = 10  # 示例值，需根据实际情况调整

    # 初始化计数器
    success5_count = 0
    success10_count = 0
    total_count = 0

    '''Test'''
    logger.info('Start test...')
    with torch.no_grad():
        mean_cd = []
        mean_hudu = []
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
            error = pose_model.module.test_metric(pc_aug, pc_norm, label)

            # CD
            cd = error['CD']
            mean_cd.append(cd.item())
            # RRE
            hudu = torch.mean(error['RRE_hd'])
            mean_hudu.append(hudu.item())
            # RR
            jiaodu = error['RRE_jd']
            success5_batch = (jiaodu < ROTATION_THRESHOLD5).sum().item()
            success10_batch = (jiaodu < ROTATION_THRESHOLD10).sum().item()
            # 更新计数器
            success5_count += success5_batch
            success10_count += success10_batch
            total_count += jiaodu.size(0)  # 或者 len(error)

    # 计算平均CD
    CD = sum(mean_cd)/len(mean_cd)
    # 计算平均RRE (弧度)
    RRE_rad = sum(mean_hudu)/len( mean_hudu)
    RRE_ang = RRE_rad * 180 / math.pi
    # 计算RR (%)
    RR5 = success5_count / total_count
    RR10 = success10_count / total_count

    csv_file = os.path.join(args.checkpoint_dir,'evaluation_metrics.csv')
    headers = ['Category', 'CD',  'RRE_ang', 'RR5', 'RR10']
    row = [args.category, CD, RRE_ang, RR5, RR10]

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
        categorys = ['bench','bench', 'bench',
                     'cabinet', 'cabinet', 'cabinet',
                     'car', 'car','car',
                     'cellphone','cellphone','cellphone',
                     'chair',  'chair', 'chair',
                     'couch', 'couch', 'couch',
                     'firearm','firearm','firearm',
                     'lamp', 'lamp', 'lamp',
                     'monitor', 'monitor','monitor',
                     'plane','plane','plane',
                     'speaker','speaker', 'speaker',
                     'table','table','table',
                     'watercraft','watercraft','watercraft']

        #categorys = ['lamp', 'speaker', 'table', 'cabinet', 'chair', 'monitor', 'bench',
        #             'watercraft', 'couch', 'plane', 'cellphone', 'firearm', 'car']
        #categorys = [ 'lamp','table',  'chair',
        #              'plane',  'cellphone', 'car']
    elif args.data_choice == 'modelnet':
        categorys = ['all','all','all']  # 训练在modelnet的所有类别
    else :
        categorys = ['ScanobjectNN','ScanobjectNN','ScanobjectNN','ScanobjectNN','ScanobjectNN',]

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
        elif args.data_choice == 'modelnet':
            TEST_DATASET = ModelNetDataLoader(root=args.modelnet_path, npoint=args.num_point, split='test',
                                              normal_channel=args.normal, category_choice=args.category)
            testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=4)
        elif args.data_choice == 'scanobject':
            TEST_DATASET = ScanObjectNNH5(data_path=args.scanobject_path, mode='test', category=args.category,
                                            num_pts=args.num_point)
            testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=8, drop_last=False)

        '''MODEL LOADING'''
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        pose_model = Network(cfg=args).cuda()
        pose_model = nn.DataParallel(pose_model, device_ids=[0])

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

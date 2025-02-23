import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
from data_utils.ScanObjectNN_Pose import ScanObjectNNH5
from data_utils.ShapeNetH5Dataloader import ShapeNetH5Loader
from data_utils.ModelNetDataLoader_Pose import ModelNetDataLoader
from data_utils.threedmatch import ThreedmatchDataset
import math
import argparse
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

    # Training
    parser.add_argument('--use_checkpoint', default=False)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size in training [default: 32]')
    parser.add_argument('--epoch', default=200, type=int, help='Number of epoch in training [default: 250]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training [default: SGD]')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate (for SGD it is multiplied by 100) [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Decay rate [default: 1e-4]')

    # Setting
    parser.add_argument('--rot_train', type=str, default='so3', choices=['aligned', 'z', 'so3'])
    parser.add_argument('--rot_test', type=str, default='so3', choices=['aligned', 'z', 'so3'])

    # Dataset
    parser.add_argument('--data_choice', default='shapenet', choices=['shapenet', 'modelnet', 'scanobject', 'threedmatch'])
    parser.add_argument('--shapenet_path', default='/home/hanbing/datasets/ShapeNetAtlasNetH5_1024')
    parser.add_argument('--modelnet_path', default='/home/hanbing/datasets/modelnet40_normal_resampled')
    parser.add_argument('--scanobject_path', default='/home/hanbing/datasets/ScanObjectNN')
    parser.add_argument('--threedmatch_path', default='/home/hanbing/datasets/3Dmatch')
    parser.add_argument('--category', type=str, default='plane')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()


def train(trainDataLoader, testDataLoader, pose_model, logger, start_epoch, log_dir, checkpoints_dir, args):

    def log_string(str):
        logger.info(str)
        print(str)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_num_batch = len(trainDataLoader)  # batchsize * train_num_batch = train data num of one epoch
    test_num_batch = len(testDataLoader)  # batchsize * val_num_batch = val data num of one epoch

    '''Optimizer init'''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            pose_model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(
            pose_model.parameters(),
            lr=args.learning_rate * 100,
            momentum=0.9,
            weight_decay=args.decay_rate
        )

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    best_test_error = 4.0

    # use TensorBoard
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    test_writer = SummaryWriter(os.path.join(log_dir, 'test'))

    '''Training'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch(%d/%s):' % ( epoch + 1, args.epoch))
        for train_batch_ind, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):

            if args.data_choice != 'threedmatch':
                pc_norm = data['pc_norm']
                if args.rot_train == 'z':
                    pc_aug = data['pc_z']
                    label = data['target_z']
                elif args.rot_train == 'so3':
                    pc_aug = data['pc_so3']
                    label = data['target_so3']
            else:
                pc_norm = data['pc_aug0']
                pc_aug = data['pc_aug1']
                label = data['R_rela']

            pc_norm = pc_norm.to(device).float()
            pc_aug = pc_aug.to(device).float()
            label = label.to(device).float()

            pose_model = pose_model.train()
            # 并行
            error = pose_model.module.training_step(pc_aug, pc_norm, label)
            # 非并行
            #error = pose_model.training_step(pc_aug, pc_norm, label)
            loss = error['loss']

            # optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

            # tensorboard
            train_step = epoch * train_num_batch + train_batch_ind
            for key, value in error.items():
                train_writer.add_scalar(f'{key}', value.item(), train_step)
            # log string
            if train_batch_ind % 102 == 0 and args.disentangle:
                print("train_angle:", error['angle_loss'].detach().cpu().numpy(),
                      "train_cd:",  error['sl_loss'].detach().cpu().numpy(),
                      "train_recons:", error['recons_loss'].detach().cpu().numpy())

            elif train_batch_ind % 102 == 0 and not args.disentangle:
                print("train_angle:", error['angle_loss'].detach().cpu().numpy()
                      #"train_cd:", error['sl_loss'].detach().cpu().numpy()
                        )


        with torch.no_grad():
            for test_batch_ind, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                if args.data_choice != 'threedmatch':
                    pc_norm = data['pc_norm']
                    pc_aug = data['pc_so3']
                    label = data['target_so3']
                else:
                    pc_norm = data['pc_aug0']
                    pc_aug = data['pc_aug1']
                    label = data['R_rela']


                pc_norm = pc_norm.to(device).float()
                pc_aug = pc_aug.to(device).float()
                label = label.to(device).float()

                pose_model = pose_model.eval()
                # 并行
                error = pose_model.module.test_step(pc_aug, pc_norm, label)
                # 非并行
                #error = pose_model.test_step(pc_aug, pc_norm, label)
                metric = error['loss']

                # tensorboard
                test_fraction_done = (test_batch_ind + 1) / test_num_batch  # 一次epoch的进度
                test_step = (epoch + test_fraction_done) * test_num_batch - 1  # 截止到目前为止的总step数
                for key, value in error.items():
                    test_writer.add_scalar(f'Metric_{key}', value.item(), test_step)
                jiaodu = error['angle_loss'].item() * 180 /math.pi
                test_writer.add_scalar('Metric_jiaodu', jiaodu, test_step)

                # log string
                if test_batch_ind % 26 == 0 and not args.disentangle:
                    print("test_angle:", error['angle_loss'].detach().cpu().numpy(),
                          #"test_cd:", error['sl_loss'].detach().cpu().numpy()
                          )

                elif test_batch_ind % 26 == 0 and args.disentangle:
                    print("train_angle:", error['angle_loss'].detach().cpu().numpy(),
                          "train_cd:", error['sl_loss'].detach().cpu().numpy(),
                          "train_recons:", error['recons_loss'].detach().cpu().numpy())

                if (metric <= best_test_error):
                    best_test_error = metric
                    best_epoch = epoch + 1

                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    state = {
                        'epoch': best_epoch,
                        'test_error': metric,
                        'model_state_dict': pose_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
    logger.info('End of training...')

def main(args, timestr):
    def log_string(str):
        logger.info(str)
        print(str)

    if args.data_choice == 'shapenet':
        categorys = ['bench', 'cabinet', 'car', 'cellphone', 'chair', 'couch', 'firearm',
                     'lamp', 'monitor', 'plane', 'speaker', 'table', 'watercraft']
        #categorys = ['car',  'chair', 'lamp', 'plane',  'table',  'cellphone']

    elif args.data_choice == 'modelnet':
        categorys = ['all']  # 训练在modelnet的所有类别
    elif args.data_choice == 'scanobject':
        categorys = ['ScanobjectNN']
    else:
        categorys = ['threedmatch']

    for i in range(len(categorys)):
        args.category = categorys[i]

        '''LOG'''
        logger = logging.getLogger("Model")
        logger.setLevel(logging.INFO)

        '''DATA LOADING'''
        if args.data_choice == 'shapenet':
            TRAIN_DATASET = ShapeNetH5Loader(data_path=args.shapenet_path, mode='train', category=args.category,
                                             num_pts=args.num_point)
            TEST_DATASET = ShapeNetH5Loader(data_path=args.shapenet_path, mode='val', category=args.category,
                                            num_pts=args.num_point)
            trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                          num_workers=4, drop_last=True)
            testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=4, drop_last=True)
        elif args.data_choice == 'modelnet':
            TRAIN_DATASET = ModelNetDataLoader(root=args.modelnet_path, npoint=args.num_point, split='train',
                                               normal_channel=args.normal, category_choice=args.category)
            TEST_DATASET = ModelNetDataLoader(root=args.modelnet_path, npoint=args.num_point, split='test',
                                              normal_channel=args.normal, category_choice=args.category)
            trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                          num_workers=4)
            testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=4)
        elif args.data_choice == 'scanobject':
            TRAIN_DATASET = ScanObjectNNH5(data_path=args.scanobject_path, mode='train', category=args.category,
                                             num_pts=args.num_point)
            TEST_DATASET = ScanObjectNNH5(data_path=args.scanobject_path, mode='test', category=args.category,
                                            num_pts=args.num_point)
            trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                          num_workers=8, drop_last=False)
            testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=8, drop_last=False)
        else:
            TRAIN_DATASET = ThreedmatchDataset(data_path=args.threedmatch_path, split='train', OVERLAP_RATIO=0.3,
                                           point_limit = 15000)
            TEST_DATASET = ThreedmatchDataset(data_path=args.threedmatch_path, split='test', OVERLAP_RATIO=0.3,
                                          point_limit = 15000)
            trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                          num_workers=8, drop_last=False)
            testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=8, drop_last=False)


        '''CREATE DIR'''
        experiment_dir = Path('./log/')
        experiment_dir = experiment_dir.joinpath(args.mode)
        experiment_dir = experiment_dir.joinpath(args.model)
        experiment_dir = experiment_dir.joinpath(f'{args.rot_train}_{args.rot_test}')
        experiment_dir = experiment_dir.joinpath(timestr)
        experiment_dir = experiment_dir.joinpath(args.category)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoints_dir = experiment_dir.joinpath('checkpoints/')
        save_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        log_dir = experiment_dir.joinpath('tb_logs/')
        log_dir.mkdir(parents=True, exist_ok=True)


        '''MODEL LOADING'''
        pose_model = Network(cfg=args)
        device_ids = [0, 1]  # 对应的 GPU 设备ID
        pose_model = nn.DataParallel(pose_model, device_ids=device_ids).cuda()

        # 计算模型大小（以 MB 为单位）
        '''parameter_count = sum(p.numel() for p in pose_model.parameters() if p.requires_grad)
        dtype_size = 4
        parameter_count = parameter_count * dtype_size / (1024 ** 2)
        print('count_parameters',parameter_count)'''

        '''CheckPoints LOADING'''
        start_epoch = 0
        if args.use_checkpoint:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            if isinstance(pose_model, torch.nn.DataParallel) or isinstance(pose_model,
                                                                           torch.nn.parallel.DistributedDataParallel):
                pose_model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                pose_model.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrain model')

        log_string(args)
        train(trainDataLoader, testDataLoader, pose_model, logger, start_epoch, log_dir, save_checkpoints_dir, args)

if __name__ == '__main__':
    args = parse_args()
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    print('被发现的GPU数量：',torch.cuda.device_count())
    main(args, timestr)
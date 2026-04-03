import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import cv2
import albumentations as A
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
# 只用deepCrack的微调官方的训练模型
'''


def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    # opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    # <-- 修改：定义数据增强 pipeline -->
    transform = None
    if opt['datasets'].get('train') is not None: # 检查是否有训练集配置
        # 注意：必须同时增强 LR 和 HR 图像
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_REFLECT),
            # 可选：颜色和噪声增强
            # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            # A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        ],
            additional_targets={'image_hr': 'image'},is_check_shapes=False) # 关键：为 HR 图像指定额外目标

    # -----------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # <-- 修改：传递 transform -->
            train_set = define_Dataset(dataset_opt, transform=transform)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)
        elif phase == 'test':
            # 测试集通常不进行增强
            test_set = define_Dataset(dataset_opt) # 不传递 transform
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)
    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    model = define_Model(opt)
    model.init_train() # 确保 model_plain.py 中的 init_train 已修改
    if opt['rank'] == 0:
        logger.info("========================================")
        logger.info("检查所有可训练的参数 (Checking all trainable parameters):")
        trainable_param_count = 0
        for name, param in model.netG.named_parameters():
            if param.requires_grad:
                logger.info(f"  --> {name}")
                trainable_param_count += 1
        if trainable_param_count == 0:
            logger.warning("警告：没有找到任何可训练的参数！模型将不会学习。")
        logger.info("========================================")
        # logger.info(model.info_network())
        # logger.info(model.info_params())
    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    for epoch in range(1000000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)
        for i, train_data in enumerate(train_loader):
            current_step += 1
            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)
            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)
            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step) # 确保支持新损失函数
            # -------------------------------
            # 4) training information
            # -------------------------------
            # 科学计数法
            # if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
            #     logs = model.current_log()  # such as loss
            #     # <-- 修改：格式化日志信息 -->
            #     message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
            #     for k, v in logs.items():  # merge log information into message
            #         message += '{:s}: {:.3e} '.format(k, v)
            #     logger.info(message)
            # 小数形式
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                # 将学习率 lr 也改为小数点形式，例如 {:.6f}
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.6f}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    # 将 {:.3e} 改为 {:.4f}，保留4位小数
                    message += '{:s}: {:.4f} '.format(k, v)
                logger.info(message)
            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)
            # -------------------------------
            # 6) testing (重点修改这里)
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                # <-- 修改：初始化平均指标 -->
                avg_psnr = 0.0
                avg_ssim = 0.0 # 新增
                avg_edge_psnr = 0.0 # 新增
                avg_continuity = 0.0 # 新增
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)
                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)
                    model.feed_data(test_data)
                    model.test()
                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E']) # 预测图像 (uint8, RGB)
                    H_img = util.tensor2uint(visuals['H']) # 真实图像 (uint8, RGB)
                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)
                    # -----------------------
                    # calculate PSNR, SSIM, Edge-PSNR, Continuity (新增指标)
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    # <-- 新增：计算 SSIM -->
                    current_ssim = util.calculate_ssim(E_img, H_img, border=border)
                    # <-- 新增：计算 Edge-PSNR -->
                    current_edge_psnr = util.calculate_edge_psnr(E_img, H_img, border=border)
                    # <-- 新增：计算 Crack Continuity -->
                    # current_continuity = util.calculate_crack_continuity(E_img, H_img)
                    current_continuity = 0.0

                    # <-- 修改：打印所有指标 -->
                    logger.info('{:->4d}--> {:>10s} | PSNR: {:<4.2f}dB | SSIM: {:.3f} | Edge-PSNR: {:.2f} | Continuity: {:.1f}%'.format(
                        idx, image_name_ext, current_psnr, current_ssim, current_edge_psnr, current_continuity*100))
                    # logger.info('{:->4d}--> {:>10s} | PSNR: {:<4.2f}dB | SSIM: {:.3f} | Edge-PSNR: {:.2f}')

                    avg_psnr += current_psnr
                    avg_ssim += current_ssim # 新增
                    avg_edge_psnr += current_edge_psnr # 新增
                    avg_continuity += current_continuity # 新增
                # <-- 修改：计算并打印平均指标 -->
                avg_psnr /= idx
                avg_ssim /= idx # 新增
                avg_edge_psnr /= idx # 新增
                avg_continuity /= idx # 新增
                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}> Average PSNR: {:<.2f}dB | SSIM: {:.3f} | Edge-PSNR: {:.2f} | Continuity: {:.1f}%'.format(
                    epoch, current_step, avg_psnr, avg_ssim, avg_edge_psnr, avg_continuity*100))
                # ----------------------------------------
                # （可选）将指标写入 Tensorboard 或文件
                # ----------------------------------------
                # 例如，如果使用 Tensorboard:
                # if opt['rank'] == 0:
                #     tb_logger.add_scalar('PSNR/test', avg_psnr, current_step)
                #     tb_logger.add_scalar('SSIM/test', avg_ssim, current_step)
                #     tb_logger.add_scalar('Edge-PSNR/test', avg_edge_psnr, current_step)
                #     tb_logger.add_scalar('Continuity/test', avg_continuity, current_step)


    # 原始代码
    # '''
    # # ----------------------------------------
    # # Step--2 (creat dataloader)
    # # ----------------------------------------
    # '''
    #
    # # ----------------------------------------
    # # 1) create_dataset
    # # 2) creat_dataloader for train and test
    # # ----------------------------------------
    # for phase, dataset_opt in opt['datasets'].items():
    #     if phase == 'train':
    #         train_set = define_Dataset(dataset_opt)
    #         train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
    #         if opt['rank'] == 0:
    #             logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
    #         if opt['dist']:
    #             train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
    #             train_loader = DataLoader(train_set,
    #                                       batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
    #                                       shuffle=False,
    #                                       num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
    #                                       drop_last=True,
    #                                       pin_memory=True,
    #                                       sampler=train_sampler)
    #         else:
    #             train_loader = DataLoader(train_set,
    #                                       batch_size=dataset_opt['dataloader_batch_size'],
    #                                       shuffle=dataset_opt['dataloader_shuffle'],
    #                                       num_workers=dataset_opt['dataloader_num_workers'],
    #                                       drop_last=True,
    #                                       pin_memory=True)
    #
    #     elif phase == 'test':
    #         test_set = define_Dataset(dataset_opt)
    #         test_loader = DataLoader(test_set, batch_size=1,
    #                                  shuffle=False, num_workers=1,
    #                                  drop_last=False, pin_memory=True)
    #     else:
    #         raise NotImplementedError("Phase [%s] is not recognized." % phase)
    #
    # '''
    # # ----------------------------------------
    # # Step--3 (initialize model)
    # # ----------------------------------------
    # '''
    #
    # model = define_Model(opt)
    # model.init_train()
    # if opt['rank'] == 0:
    #     logger.info(model.info_network())
    #     logger.info(model.info_params())
    #
    # '''
    # # ----------------------------------------
    # # Step--4 (main training)
    # # ----------------------------------------
    # '''
    #
    # for epoch in range(1000000):  # keep running
    #     if opt['dist']:
    #         train_sampler.set_epoch(epoch + seed)
    #
    #     for i, train_data in enumerate(train_loader):
    #
    #         current_step += 1
    #
    #         # -------------------------------
    #         # 1) update learning rate
    #         # -------------------------------
    #         model.update_learning_rate(current_step)
    #
    #         # -------------------------------
    #         # 2) feed patch pairs
    #         # -------------------------------
    #         model.feed_data(train_data)
    #
    #         # -------------------------------
    #         # 3) optimize parameters
    #         # -------------------------------
    #         model.optimize_parameters(current_step)
    #
    #         # -------------------------------
    #         # 4) training information
    #         # -------------------------------
    #         if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
    #             logs = model.current_log()  # such as loss
    #             message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
    #             for k, v in logs.items():  # merge log information into message
    #                 message += '{:s}: {:.3e} '.format(k, v)
    #             logger.info(message)
    #
    #         # -------------------------------
    #         # 5) save model
    #         # -------------------------------
    #         if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
    #             logger.info('Saving the model.')
    #             model.save(current_step)
    #
    #         # -------------------------------
    #         # 6) testing
    #         # -------------------------------
    #         if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
    #
    #             avg_psnr = 0.0
    #             idx = 0
    #
    #             for test_data in test_loader:
    #                 idx += 1
    #                 image_name_ext = os.path.basename(test_data['L_path'][0])
    #                 img_name, ext = os.path.splitext(image_name_ext)
    #
    #                 img_dir = os.path.join(opt['path']['images'], img_name)
    #                 util.mkdir(img_dir)
    #
    #                 model.feed_data(test_data)
    #                 model.test()
    #
    #                 visuals = model.current_visuals()
    #                 E_img = util.tensor2uint(visuals['E'])
    #                 H_img = util.tensor2uint(visuals['H'])
    #
    #                 # -----------------------
    #                 # save estimated image E
    #                 # -----------------------
    #                 save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
    #                 util.imsave(E_img, save_img_path)
    #
    #                 # -----------------------
    #                 # calculate PSNR
    #                 # -----------------------
    #                 current_psnr = util.calculate_psnr(E_img, H_img, border=border)
    #
    #                 logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))
    #
    #                 avg_psnr += current_psnr
    #
    #             avg_psnr = avg_psnr / idx
    #
    #             # testing log
    #             logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

if __name__ == '__main__':
    main()

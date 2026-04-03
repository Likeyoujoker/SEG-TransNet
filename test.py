import os
import argparse
import logging
import numpy as np
from torch.utils.data import DataLoader

from utils.utils_image import calculate_edge_psnr

try:
    from utils import utils_logger
    from utils import utils_image as util
    from utils import utils_option as option
    from utils.utils_dist import get_dist_info, init_dist
    from data.select_dataset import define_Dataset
    from models.select_model import define_Model
except ImportError:
    print('Error: please run this script from the repository root with the required packages installed.')
    raise


def main_test(json_path, model_path, save_suffix='test_results'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_y_channel', type=bool, default=True, help='If True, test on Y channel.')
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--model_path', type=str, default=model_path, help='Path to the trained G model file.')
    parser.add_argument('--save_suffix', type=str, default=save_suffix, help='Subfolder name to save images.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)
    opt['dist'] = args.dist

    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    opt['path']['pretrained_netG'] = args.model_path
    border = opt.get('scale', 4)

    if opt['rank'] == 0:
        logger_name = 'test'
        log_path = os.path.join(opt['path']['log'], f'{logger_name}_{args.save_suffix}.log')
        utils_logger.logger_info(logger_name, log_path)
        logger = logging.getLogger(logger_name)
        logger.info(f'Start evaluation: {args.model_path}')
        logger.info(option.dict2str(opt))

    test_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

    if test_loader is None:
        logger.error("No 'test' dataset was found in the config.")
        return

    model = define_Model(opt)
    model.load()

    if opt['rank'] == 0:
        logger.info(f'Model loaded: {opt["path"]["pretrained_netG"]}')
        logger.info(f'Evaluating on {opt["datasets"]["test"]["name"]}')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_edge_psnr = 0.0
        idx = 0
        test_img_save_dir = os.path.join(opt['path']['images'], args.save_suffix)
        util.mkdir(test_img_save_dir)
        logger.info(f'Results will be saved to {test_img_save_dir}')
        for test_data in test_loader:
            idx += 1
            image_name_ext = os.path.basename(test_data['L_path'][0])
            img_name, ext = os.path.splitext(image_name_ext)
            model.feed_data(test_data)
            model.test()
            visuals = model.current_visuals()
            e_img = util.tensor2uint(visuals['E'])
            h_img = util.tensor2uint(visuals['H'])
            if getattr(args, 'test_y_channel', True):
                calc_e = np.round(util.rgb2ycbcr(e_img, only_y=True)).astype(np.uint8)
                calc_h = np.round(util.rgb2ycbcr(h_img, only_y=True)).astype(np.uint8)
                channel_msg = 'Y-Channel'
            else:
                calc_e = e_img
                calc_h = h_img
                channel_msg = 'RGB'
            save_img_path = os.path.join(test_img_save_dir, f'{img_name}_SR{ext}')
            util.imsave(e_img, save_img_path)
            current_psnr = util.calculate_psnr(calc_e, calc_h, border=border)
            current_ssim = util.calculate_ssim(calc_e, calc_h, border=border)
            current_edge_psnr = calculate_edge_psnr(calc_e, calc_h, border=border)
            logger.info('{:->4d}--> {:>10s} | Mode: {} | PSNR: {:<4.2f}dB | SSIM: {:.3f} | Edge-PSNR: {:.2f}'.format(idx, image_name_ext, channel_msg, current_psnr, current_ssim, current_edge_psnr))
            avg_psnr += current_psnr
            avg_ssim += current_ssim
            avg_edge_psnr += current_edge_psnr
        avg_psnr /= idx
        avg_ssim /= idx
        avg_edge_psnr /= idx
        logger.info('<-- Evaluation finished -->')
        logger.info('Average PSNR: {:<.2f}dB'.format(avg_psnr))
        logger.info('Average SSIM: {:.3f}'.format(avg_ssim))
        logger.info('Average Edge-PSNR: {:.2f}'.format(avg_edge_psnr))


if __name__ == '__main__':
    main_test(json_path='options/test_seg_paper.json', model_path='weights/your_trained_model.pth', save_suffix='seg_eval')

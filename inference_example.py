import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.select_model import define_Model
from utils import utils_image as util
from utils import utils_option as option


def load_model(opt_path: str, model_path: str):
    opt = option.parse(opt_path, is_train=False)
    opt['path']['pretrained_netG'] = model_path
    model = define_Model(opt)
    model.load()
    return model, opt


def read_input_paths(input_path: str):
    if os.path.isdir(input_path):
        return util.get_image_paths(input_path)
    if os.path.isfile(input_path):
        return [input_path]
    raise FileNotFoundError(f'Input path not found: {input_path}')


def prepare_tensor(image_path: str, device: torch.device, window_size: int):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_chw = np.transpose(image_rgb, (2, 0, 1))
    tensor = torch.from_numpy(image_chw).float().unsqueeze(0).to(device)

    _, _, height, width = tensor.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, height, width


def save_result(output_tensor: torch.Tensor, output_path: str, height: int, width: int, scale: int):
    output = output_tensor[:, :, :height * scale, :width * scale]
    output = output.squeeze(0).clamp_(0, 1).cpu().numpy()
    output = np.transpose(output, (1, 2, 0))
    output = cv2.cvtColor((output * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference example for the public release package.')
    parser.add_argument('--opt', type=str, default='options/test_seg_paper.json', help='Path to the option JSON file.')
    parser.add_argument('--model_path', type=str, default='weights/your_trained_model.pth', help='Path to the trained model checkpoint.')
    parser.add_argument('--input_path', type=str, default='datasets/Crack_test_x4/LR_bicubic', help='Input image file or directory.')
    parser.add_argument('--output_dir', type=str, default='results/demo', help='Directory to store SR outputs.')
    args = parser.parse_args()

    model, opt = load_model(args.opt, args.model_path)
    device = next(model.netG.parameters()).device
    window_size = opt['netG'].get('window_size', 8)
    scale = opt['scale']
    util.mkdirs([args.output_dir])

    for image_path in read_input_paths(args.input_path):
        tensor, height, width = prepare_tensor(image_path, device, window_size)
        model.feed_data({'L': tensor}, need_H=False)
        model.test()
        output_tensor = model.current_visuals(need_H=False)['E']
        output_path = os.path.join(args.output_dir, f'{Path(image_path).stem}_SR.png')
        save_result(output_tensor, output_path, height, width, scale)
        print(f'Saved {output_path}')

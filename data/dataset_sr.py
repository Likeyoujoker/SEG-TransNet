import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import cv2

class DatasetSR(data.Dataset):
    def __init__(self, opt, transform=None): # <-- 修改：添加 transform 参数
        """初始化 DatasetSR。

        Args:
            opt (dict): 配置选项。
            transform (callable, optional): 一个可调用的变换对象（例如，
                albumentations.Compose 对象），用于在获取数据项时应用数据增强。
                Defaults to None.
        """
        super(DatasetSR, self).__init__()
        self.opt = opt
        # <-- 修改：保存 transform -->
        self.transform = transform
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])
        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):
        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        # === 新增：调试代码 ===
        if img_H is None:
            print(f"\n[Error] 无法读取高分辨率图片 (GT): {H_path}")
            print(f"请检查该路径是否存在，或文件是否损坏。")
            # 也可以在这里检查 LR 图片
            if self.paths_L:
                L_path = self.paths_L[index]
                print(f"[Info] 对应的低分辨率图片 (LR) 路径是: {L_path}")
            raise ValueError(f"图片读取失败: {H_path}")
        # ====================
        img_H = util.uint2single(img_H)
        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)
        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, self.n_channels)
            img_L = util.uint2single(img_L)
        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':
            H, W, C = img_L.shape
            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            # <-- 修改：注释掉或移除旧的增强方式，因为我们将使用 Albumentations -->
            # mode = random.randint(0, 7)
            # img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)
            # --------------------------------
            # Apply Albumentations Transform
            # --------------------------------
            # <-- 新增：应用 Albumentations 增强 -->
            if self.transform is not None:
                # Albumentations 期望输入是 [H, W, C] 的 numpy 数组 (通常是 uint8)
                # 但我们当前是 float [0, 1]。需要临时转换。
                # 注意：如果你的数据已经是 uint8 格式，则不需要转换。
                # 假设 img_L 和 img_H 是 float32 [0, 1]
                img_L_uint8 = (img_L * 255).astype(np.uint8)
                img_H_uint8 = (img_H * 255).astype(np.uint8)

                # 应用增强。需要指定 HR 图像作为额外目标
                # 确保在定义 transform 时使用了 additional_targets={'image_hr': 'image'}
                transformed = self.transform(image=img_L_uint8, image_hr=img_H_uint8)
                img_L_aug = transformed['image']
                img_H_aug = transformed['image_hr']

                # 转换回 float32 [0, 1]
                img_L = img_L_aug.astype(np.float32) / 255.0
                img_H = img_H_aug.astype(np.float32) / 255.0
            # <-- 新增结束 -->

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        # <-- 修改：确保 img_H 和 img_L 是 float32 -->
        img_H = img_H.astype(np.float32)
        img_L = img_L.astype(np.float32)
        img_H = torch.from_numpy(np.transpose(img_H, (2, 0, 1))).contiguous()
        img_L = torch.from_numpy(np.transpose(img_L, (2, 0, 1))).contiguous()
        # img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        if L_path is None:
            L_path = H_path
        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    # 原始代码
    # '''
    # # -----------------------------------------
    # # Get L/H for SISR.
    # # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # # -----------------------------------------
    # # e.g., SRResNet
    # # -----------------------------------------
    # '''
    #
    # def __init__(self, opt):
    #     super(DatasetSR, self).__init__()
    #     self.opt = opt
    #     self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
    #     self.sf = opt['scale'] if opt['scale'] else 4
    #     self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
    #     self.L_size = self.patch_size // self.sf
    #
    #     # ------------------------------------
    #     # get paths of L/H
    #     # ------------------------------------
    #     self.paths_H = util.get_image_paths(opt['dataroot_H'])
    #     self.paths_L = util.get_image_paths(opt['dataroot_L'])
    #
    #     assert self.paths_H, 'Error: H path is empty.'
    #     if self.paths_L and self.paths_H:
    #         assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))
    #
    # def __getitem__(self, index):
    #
    #     L_path = None
    #     # ------------------------------------
    #     # get H image
    #     # ------------------------------------
    #     H_path = self.paths_H[index]
    #     img_H = util.imread_uint(H_path, self.n_channels)
    #     img_H = util.uint2single(img_H)
    #
    #     # ------------------------------------
    #     # modcrop
    #     # ------------------------------------
    #     img_H = util.modcrop(img_H, self.sf)
    #
    #     # ------------------------------------
    #     # get L image
    #     # ------------------------------------
    #     if self.paths_L:
    #         # --------------------------------
    #         # directly load L image
    #         # --------------------------------
    #         L_path = self.paths_L[index]
    #         img_L = util.imread_uint(L_path, self.n_channels)
    #         img_L = util.uint2single(img_L)
    #
    #     else:
    #         # --------------------------------
    #         # sythesize L image via matlab's bicubic
    #         # --------------------------------
    #         H, W = img_H.shape[:2]
    #         img_L = util.imresize_np(img_H, 1 / self.sf, True)
    #
    #     # ------------------------------------
    #     # if train, get L/H patch pair
    #     # ------------------------------------
    #     if self.opt['phase'] == 'train':
    #
    #         H, W, C = img_L.shape
    #
    #         # --------------------------------
    #         # randomly crop the L patch
    #         # --------------------------------
    #         rnd_h = random.randint(0, max(0, H - self.L_size))
    #         rnd_w = random.randint(0, max(0, W - self.L_size))
    #         img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
    #
    #         # --------------------------------
    #         # crop corresponding H patch
    #         # --------------------------------
    #         rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
    #         img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
    #
    #         # --------------------------------
    #         # augmentation - flip and/or rotate
    #         # --------------------------------
    #         mode = random.randint(0, 7)
    #         img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)
    #
    #     # ------------------------------------
    #     # L/H pairs, HWC to CHW, numpy to tensor
    #     # ------------------------------------
    #     img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
    #
    #     if L_path is None:
    #         L_path = H_path
    #
    #     return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)

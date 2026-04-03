
import albumentations as A
import cv2
from cv2 import transform

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt, transform=None):    #原始参数没有transform
    """根据配置选择并创建数据集实例。

    Args:
        dataset_opt (dict): 数据集的配置选项。
        transform (callable, optional): 一个可调用的变换对象（例如，
            albumentations.Compose 对象），用于在获取数据项时应用数据增强。
            Defaults to None.

    Returns:
        torch.utils.data.Dataset: 创建的数据集实例。
    """
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type in ['l', 'low-quality', 'input-only']:
        from data.dataset_l import DatasetL as D
    # -----------------------------------------
    # denoising
    # -----------------------------------------
    elif dataset_type in ['dncnn', 'denoising']:
        from data.dataset_dncnn import DatasetDnCNN as D
    elif dataset_type in ['dnpatch']:
        from data.dataset_dnpatch import DatasetDnPatch as D
    elif dataset_type in ['ffdnet', 'denoising-noiselevel']:
        from data.dataset_ffdnet import DatasetFFDNet as D
    elif dataset_type in ['fdncnn', 'denoising-noiselevelmap']:
        from data.dataset_fdncnn import DatasetFDnCNN as D
    # -----------------------------------------
    # super-resolution (重点修改这里)
    # -----------------------------------------
    elif dataset_type in ['sr', 'super-resolution']:
        from data.dataset_sr import DatasetSR as D
        # <-- 修改：在创建 DatasetSR 时传递 transform -->
        dataset = D(dataset_opt, transform=transform)
        print('Dataset [{:s} - {:s}] is created with augmentation.'.format(dataset.__class__.__name__, dataset_opt['name']))
        return dataset
    elif dataset_type in ['srmd']:
        from data.dataset_srmd import DatasetSRMD as D
    elif dataset_type in ['dpsr', 'dnsr']:
        from data.dataset_dpsr import DatasetDPSR as D
    elif dataset_type in ['usrnet', 'usrgan']:
        from data.dataset_usrnet import DatasetUSRNet as D
    elif dataset_type in ['bsrnet', 'bsrgan', 'blindsr']:
        from data.dataset_blindsr import DatasetBlindSR as D
    # -------------------------------------------------
    # JPEG compression artifact reduction (deblocking)
    # -------------------------------------------------
    elif dataset_type in ['jpeg']:
        from data.dataset_jpeg import DatasetJPEG as D
    # -----------------------------------------
    # video restoration
    # -----------------------------------------
    elif dataset_type in ['videorecurrenttraindataset']:
        from data.dataset_video_train import VideoRecurrentTrainDataset as D
    elif dataset_type in ['videorecurrenttrainnonblinddenoisingdataset']:
        from data.dataset_video_train import VideoRecurrentTrainNonblindDenoisingDataset as D
    elif dataset_type in ['videorecurrenttrainvimeodataset']:
        from data.dataset_video_train import VideoRecurrentTrainVimeoDataset as D
    elif dataset_type in ['videorecurrenttrainvimeovfidataset']:
        from data.dataset_video_train import VideoRecurrentTrainVimeoVFIDataset as D
    elif dataset_type in ['videorecurrenttestdataset']:
        from data.dataset_video_test import VideoRecurrentTestDataset as D
    elif dataset_type in ['singlevideorecurrenttestdataset']:
        from data.dataset_video_test import SingleVideoRecurrentTestDataset as D
    elif dataset_type in ['videotestvimeo90kdataset']:
        from data.dataset_video_test import VideoTestVimeo90KDataset as D
    elif dataset_type in ['vfi_davis']:
        from data.dataset_video_test import VFI_DAVIS as D
    elif dataset_type in ['vfi_ucf101']:
        from data.dataset_video_test import VFI_UCF101 as D
    elif dataset_type in ['vfi_vid4']:
        from data.dataset_video_test import VFI_Vid4 as D
    # -----------------------------------------
    # common
    # -----------------------------------------
    elif dataset_type in ['plain']:
        from data.dataset_plain import DatasetPlain as D
    elif dataset_type in ['plainpatch']:
        from data.dataset_plainpatch import DatasetPlainPatch as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    # <-- 修改：对于非 'sr' 类型的数据集，如果需要 transform，也需要在这里处理 -->
    # 注意：下面的代码假设只有 SR 数据集需要 transform。如果不是，请相应修改其他分支。
    # 对于其他类型的数据集，我们按原始方式创建（但它们不会使用到 transform）
    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset

    # 原始代码
    # dataset_type = dataset_opt['dataset_type'].lower()
    # if dataset_type in ['l', 'low-quality', 'input-only']:
    #     from data.dataset_l import DatasetL as D
    #
    # # -----------------------------------------
    # # denoising
    # # -----------------------------------------
    # elif dataset_type in ['dncnn', 'denoising']:
    #     from data.dataset_dncnn import DatasetDnCNN as D
    #
    # elif dataset_type in ['dnpatch']:
    #     from data.dataset_dnpatch import DatasetDnPatch as D
    #
    # elif dataset_type in ['ffdnet', 'denoising-noiselevel']:
    #     from data.dataset_ffdnet import DatasetFFDNet as D
    #
    # elif dataset_type in ['fdncnn', 'denoising-noiselevelmap']:
    #     from data.dataset_fdncnn import DatasetFDnCNN as D
    #
    # # -----------------------------------------
    # # super-resolution
    # # -----------------------------------------
    # elif dataset_type in ['sr', 'super-resolution']:
    #     from data.dataset_sr import DatasetSR as D
    #
    # elif dataset_type in ['srmd']:
    #     from data.dataset_srmd import DatasetSRMD as D
    #
    # elif dataset_type in ['dpsr', 'dnsr']:
    #     from data.dataset_dpsr import DatasetDPSR as D
    #
    # elif dataset_type in ['usrnet', 'usrgan']:
    #     from data.dataset_usrnet import DatasetUSRNet as D
    #
    # elif dataset_type in ['bsrnet', 'bsrgan', 'blindsr']:
    #     from data.dataset_blindsr import DatasetBlindSR as D
    #
    # # -------------------------------------------------
    # # JPEG compression artifact reduction (deblocking)
    # # -------------------------------------------------
    # elif dataset_type in ['jpeg']:
    #     from data.dataset_jpeg import DatasetJPEG as D
    #
    # # -----------------------------------------
    # # video restoration
    # # -----------------------------------------
    # elif dataset_type in ['videorecurrenttraindataset']:
    #     from data.dataset_video_train import VideoRecurrentTrainDataset as D
    # elif dataset_type in ['videorecurrenttrainnonblinddenoisingdataset']:
    #     from data.dataset_video_train import VideoRecurrentTrainNonblindDenoisingDataset as D
    # elif dataset_type in ['videorecurrenttrainvimeodataset']:
    #     from data.dataset_video_train import VideoRecurrentTrainVimeoDataset as D
    # elif dataset_type in ['videorecurrenttrainvimeovfidataset']:
    #     from data.dataset_video_train import VideoRecurrentTrainVimeoVFIDataset as D
    # elif dataset_type in ['videorecurrenttestdataset']:
    #     from data.dataset_video_test import VideoRecurrentTestDataset as D
    # elif dataset_type in ['singlevideorecurrenttestdataset']:
    #     from data.dataset_video_test import SingleVideoRecurrentTestDataset as D
    # elif dataset_type in ['videotestvimeo90kdataset']:
    #     from data.dataset_video_test import VideoTestVimeo90KDataset as D
    # elif dataset_type in ['vfi_davis']:
    #     from data.dataset_video_test import VFI_DAVIS as D
    # elif dataset_type in ['vfi_ucf101']:
    #     from data.dataset_video_test import VFI_UCF101 as D
    # elif dataset_type in ['vfi_vid4']:
    #     from data.dataset_video_test import VFI_Vid4 as D
    #
    #
    # # -----------------------------------------
    # # common
    # # -----------------------------------------
    # elif dataset_type in ['plain']:
    #     from data.dataset_plain import DatasetPlain as D
    #
    # elif dataset_type in ['plainpatch']:
    #     from data.dataset_plainpatch import DatasetPlainPatch as D
    #
    # else:
    #     raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))
    #
    # dataset = D(dataset_opt)
    # print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    # return dataset

import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss
from utils import utils_image as util
from utils.utils_model import test_mode
import torch.nn.functional as F
from piqa import SSIM


class ModelPlain(ModelBase):
    """Plain SR training model used by the retained paper workflow."""

    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        self.opt_train = self.opt['train']
        self.cri_ssim = SSIM().to(self.device)
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    def init_train(self):
        self.load()
        self.netG.train()
        self.define_loss()
        self.define_optimizer()
        self.load_optimizers()
        self.define_scheduler()
        self.log_dict = OrderedDict()

    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    def define_loss(self):
        loss_type = self.opt_train['G_lossfn_type']
        if '+' in loss_type:
            logging.getLogger(__name__).info('Composite loss [%s] is handled in optimize_parameters.', loss_type)
        elif loss_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif loss_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif loss_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif loss_type == 'ssim':
            self.G_lossfn = self.cri_ssim
        elif loss_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train.get('G_charbonnier_eps', 1e-9)).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(loss_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    def define_optimizer(self):
        params = [v for _, v in self.netG.named_parameters() if v.requires_grad]
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(
                params,
                lr=self.opt_train['G_optimizer_lr'],
                betas=self.opt_train['G_optimizer_betas'],
                weight_decay=self.opt_train['G_optimizer_wd']
            )
        else:
            raise NotImplementedError

    def define_scheduler(self):
        self.schedulers = []
        scheduler_type = self.opt_train['G_scheduler_type']
        if scheduler_type == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(
                self.G_optimizer,
                milestones=self.opt_train['G_scheduler_milestones'],
                gamma=self.opt_train['G_scheduler_gamma']
            )
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                self.G_optimizer,
                T_max=self.opt_train['G_scheduler_T_max'],
                eta_min=self.opt_train['G_scheduler_eta_min']
            )
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented.')
        self.schedulers.append(scheduler)

    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    def netG_forward(self):
        self.E = self.netG(self.L)

    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG.train()
        self.E = self.netG(self.L)
        loss_G = 0
        loss_dict = {}
        loss_type = self.opt['train']['G_lossfn_type']
        loss_weight = self.opt['train'].get('G_lossfn_weight', 1.0)

        if '+' in loss_type:
            loss_types = loss_type.split('+')
            weights = loss_weight if isinstance(loss_weight, list) else [loss_weight] * len(loss_types)
            for lt, w in zip(loss_types, weights):
                if lt == 'l1':
                    l_val = F.l1_loss(self.E, self.H, reduction='mean')
                    loss_G += w * l_val
                    loss_dict['G_l1'] = l_val.item()
                elif lt == 'l2':
                    l_val = F.mse_loss(self.E, self.H, reduction='mean')
                    loss_G += w * l_val
                    loss_dict['G_l2'] = l_val.item()
                elif lt == 'ssim':
                    e_clamped = torch.clamp(self.E, 0.0, 1.0)
                    h_clamped = torch.clamp(self.H, 0.0, 1.0)
                    l_val = 1 - self.cri_ssim(e_clamped, h_clamped)
                    loss_G += w * l_val
                    loss_dict['G_ssim'] = l_val.item()
                elif lt == 'grad':
                    l_val = util.gradient_loss(self.E, self.H)
                    loss_G += w * l_val
                    loss_dict['G_grad'] = l_val.item()
                else:
                    raise NotImplementedError(f'Loss function type [{lt}] is not recognized.')
        else:
            if loss_type == 'l1':
                l_val = F.l1_loss(self.E, self.H, reduction='mean')
            elif loss_type == 'l2':
                l_val = F.mse_loss(self.E, self.H, reduction='mean')
            elif loss_type == 'ssim':
                l_val = 1 - self.cri_ssim(torch.clamp(self.E, 0.0, 1.0), torch.clamp(self.H, 0.0, 1.0))
            else:
                raise NotImplementedError(f'Loss function type [{loss_type}] is not recognized.')
            loss_G += l_val
            loss_dict['G_loss'] = l_val.item()

        loss_G.backward()
        clip_grad = self.opt['train'].get('G_optimizer_clipgrad', None)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), clip_grad)
        self.G_optimizer.step()
        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])
        self.log_dict = loss_dict

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    def current_log(self):
        return self.log_dict

    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    def print_network(self):
        print(self.describe_network(self.netG))

    def print_params(self):
        print(self.describe_params(self.netG))

import os
import torch
import torch.nn as nn
from utils.utils_bnorm import merge_bn, tidy_sequential
from torch.nn.parallel import DataParallel, DistributedDataParallel


class ModelBase():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt['path']['models']
        use_cuda = bool(opt.get('gpu_ids')) and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []

    def init_train(self):
        pass

    def load(self):
        pass

    def save(self, label):
        pass

    def define_loss(self):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def current_visuals(self):
        pass

    def current_losses(self):
        pass

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    def current_learning_rate(self):
        if hasattr(self.schedulers[0], 'get_last_lr'):
            return self.schedulers[0].get_last_lr()[0]
        return self.schedulers[0].get_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    def get_bare_model(self, network):
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def model_to_device(self, network):
        network = network.to(self.device)
        if self.opt['dist'] and self.device.type == 'cuda':
            find_unused_parameters = self.opt.get('find_unused_parameters', True)
            use_static_graph = self.opt.get('use_static_graph', False)
            network = DistributedDataParallel(network, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
            if use_static_graph:
                network._set_static_graph()
        elif self.device.type == 'cuda' and len(self.opt.get('gpu_ids', [])) > 1:
            network = DataParallel(network)
        return network

    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    def describe_params(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        for name, param in network.state_dict().items():
            if 'num_batches_tracked' not in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        return msg

    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=False, param_key='params'):
        network = self.get_bare_model(network)
        state_dict_loaded = torch.load(load_path, map_location=self.device)
        if isinstance(state_dict_loaded, dict) and param_key in state_dict_loaded:
            state_dict_loaded = state_dict_loaded[param_key]
        state_dict_model = network.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict_loaded.items() if k in state_dict_model and v.shape == state_dict_model[k].shape}
        state_dict_model.update(filtered_state_dict)
        network.load_state_dict(state_dict_model, strict=False)

    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def load_optimizer(self, load_path, optimizer):
        optimizer.load_state_dict(torch.load(load_path, map_location=self.device))

    def update_E(self, decay=0.999):
        netG = self.get_bare_model(self.netG)
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1 - decay)

    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)

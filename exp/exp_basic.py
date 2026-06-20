import os
import torch
from models import Autoformer, DLinear, Informer, iTransformer, TiDE, TimeMixer, TSMixer,  TimeXer, TimeBridge, CarbonMix,PatchMLP


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'Informer': Informer,
            'iTransformer': iTransformer,
            'TiDE': TiDE,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'TimeXer': TimeXer,
            'TimeBridge':TimeBridge,
            'CarbonMix': CarbonMix,
            'PatchMLP':PatchMLP,
        }
        
        self.device = self._acquire_device()
        # Build model and move to primary device first, then wrap DP if enabled
        base_model = self._build_model()
        self.model = base_model.to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # run.py already sets CUDA_VISIBLE_DEVICES before Exp is created.
            # After that, visible GPUs are remapped to cuda:0, cuda:1, ... from process view.
            # So we always use cuda:0 for single GPU, or cuda:0 as primary for multi-GPU.
            device = torch.device('cuda:0')
            if self.args.use_multi_gpu:
                print(f"Use Multi-GPU: primary cuda:0, devices [{self.args.devices}]")
            else:
                print('Use GPU: cuda:0 (physical GPU {} via CUDA_VISIBLE_DEVICES)'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

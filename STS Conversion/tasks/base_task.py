import subprocess
import traceback
from datetime import datetime
from functools import wraps
from torch.utils.tensorboard import SummaryWriter
from utils.hparams import hparams, set_hparams
import random
import sys
import numpy as np
from utils.trainer import Trainer
from torch import nn
import torch.utils.data
import utils
import logging
import os

torch.multiprocessing.set_sharing_strategy(os.getenv('TORCH_SHARE_STRATEGY', 'file_system'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def data_loader(fn):
    """
    Decorator to make any fx with this use the lazy property
    :param fn:
    :return:
    """

    wraps(fn)
    attr_name = '_lazy_' + fn.__name__

    def _get_data_loader(self):
        try:
            value = getattr(self, attr_name)
        except AttributeError:
            try:
                value = fn(self)  # Lazy evaluation, done only once.
            except AttributeError as e:
                # Guard against AttributeError suppression. (Issue #142)
                traceback.print_exc()
                error = f'{fn.__name__}: An AttributeError was encountered: ' + str(e)
                raise RuntimeError(error) from e
            setattr(self, attr_name, value)  # Memoize evaluation.
        return value

    return _get_data_loader


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle):
        super().__init__()
        self.hparams = hparams
        self.shuffle = shuffle
        self.sort_by_len = hparams['sort_by_len']
        self.sizes = None

    @property
    def _sizes(self):
        return self.sizes

    def __getitem__(self, index):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError

    def __len__(self):
        return len(self._sizes)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        size = min(self._sizes[index], hparams['max_frames'])
        return size

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', hparams['ds_workers']))


class BaseTask(nn.Module):
    def __init__(self, *args, **kwargs):
        # dataset configs
        super(BaseTask, self).__init__()
        self.current_epoch = 0
        self.global_step = 0
        self.trainer = None
        self.use_ddp = False
        self.gradient_clip_val = hparams['clip_grad_norm']
        self.model = None
        self.training_losses_meter = None
        self.logger = None

    ######################
    # build model, dataloaders, optimizer, scheduler and tensorboard
    ######################
    def build_model(self):
        raise NotImplementedError

    @data_loader
    def train_dataloader(self):
        raise NotImplementedError

    @data_loader
    def test_dataloader(self):
        raise NotImplementedError

    @data_loader
    def val_dataloader(self):
        raise NotImplementedError

    def build_scheduler(self, optimizer):
        raise NotImplementedError

    def build_optimizer(self, model):
        raise NotImplementedError

    def configure_optimizers(self):
        optm = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(optm)
        if isinstance(optm, (list, tuple)):
            return optm
        return [optm]

    def build_tensorboard(self, save_dir, name, version, **kwargs):
        root_dir = os.path.join(save_dir, name)
        os.makedirs(root_dir, exist_ok=True)
        log_dir = os.path.join(root_dir, "version_" + str(version))
        self.logger = SummaryWriter(log_dir=log_dir, **kwargs)
        self.plot_model()

    def plot_model(self):
        pass

    ######################
    # training
    ######################
    def on_train_start(self):
        pass

    def on_epoch_start(self):
        self.training_losses_meter = {'total_loss': utils.AvgrageMeter()}

    def _training_step(self, sample, batch_idx, optimizer_idx):
        """

        :param sample:
        :param batch_idx:
        :return: total loss: torch.Tensor, loss_log: dict
        """
        raise NotImplementedError

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        """

        :param sample:
        :param batch_idx:
        :param optimizer_idx:
        :return: {'loss': torch.Tensor, 'progress_bar': dict, 'tb_log': dict}
        """
        loss_ret = self._training_step(sample, batch_idx, optimizer_idx)
        self.opt_idx = optimizer_idx
        if loss_ret is None:
            return {'loss': None}
        total_loss, log_outputs = loss_ret
        log_outputs = utils.tensors_to_scalars(log_outputs)
        for k, v in log_outputs.items():
            if k not in self.training_losses_meter:
                self.training_losses_meter[k] = utils.AvgrageMeter()
            if not np.isnan(v):
                self.training_losses_meter[k].update(v)
        self.training_losses_meter['total_loss'].update(total_loss.item())

        try:
            log_outputs['lr'] = self.scheduler.get_last_lr()
            if isinstance(log_outputs['lr'], list):
                log_outputs['lr'] = log_outputs['lr'][0]
        except:
            pass

        # log_outputs['all_loss'] = total_loss.item()
        progress_bar_log = log_outputs
        tb_log = {f'tr/{k}': v for k, v in log_outputs.items()}
        return {
            'loss': total_loss,
            'progress_bar': progress_bar_log,
            'tb_log': tb_log
        }

    def on_before_optimization(self):
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    def on_epoch_end(self):
        loss_outputs = {k: round(v.avg, 4) for k, v in self.training_losses_meter.items()}
        print(f"Epoch {self.current_epoch} ended. Steps: {self.global_step}. {loss_outputs}")

    def on_train_end(self):
        pass

    ######################
    # validation
    ######################
    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: output: dict
        """
        raise NotImplementedError

    def _validation_end(self, outputs):
        """

        :param outputs:
        :return: loss_output: dict
        """
        raise NotImplementedError

    def validation_end(self, outputs):
        loss_output = self._validation_end(outputs)
        print(f"| Valid results: {loss_output}")
        return {
            'tb_log': {f'val/{k}': v for k, v in loss_output.items()},
            'val_loss': loss_output['total_loss']
        }

    ######################
    # testing
    ######################
    def test_start(self):
        pass

    def test_step(self, sample, batch_idx):
        return self.validation_step(sample, batch_idx)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    ######################
    # utils
    ######################
    def load_ckpt(self, ckpt_base_dir, current_model_name=None, model_name='model', force=True, strict=True):
        if current_model_name is None:
            current_model_name = model_name
        utils.load_ckpt(self.__getattr__(current_model_name), ckpt_base_dir, current_model_name, force, strict)

    ######################
    # start training/testing
    ######################
    def start(self):
        set_hparams()
        os.environ['MASTER_PORT'] = str(random.randint(15000, 30000))
        random.seed(hparams['seed'])
        np.random.seed(hparams['seed'])
        work_dir = hparams['work_dir']
        trainer = Trainer(
            work_dir=work_dir,
            val_check_interval=hparams['val_check_interval'],
            tb_log_interval=hparams['tb_log_interval'],
            max_updates=hparams['max_updates'],
            num_sanity_val_steps=hparams['num_sanity_val_steps'] if not hparams['validate'] else 10000,
            accumulate_grad_batches=hparams['accumulate_grad_batches'],
            print_nan_grads=hparams['print_nan_grads'],
            resume_from_checkpoint=hparams.get('resume_from_checkpoint', 0),
            amp=hparams['amp'],
            # save ckpt
            monitor_key=hparams['valid_monitor_key'],
            monitor_mode=hparams['valid_monitor_mode'],
            num_ckpt_keep=hparams['num_ckpt_keep'],
            save_best=hparams['save_best'],
            seed=hparams['seed'],
            debug=hparams['debug']
        )
        if not hparams['infer']:  # train
            t = datetime.now().strftime('%Y%m%d%H%M%S')
            code_dir = f'{work_dir}/codes/{t}'
            subprocess.check_call(f'mkdir -p "{code_dir}"', shell=True)
            for c in hparams['save_codes']:
                if os.path.exists(c):
                    subprocess.check_call(f'cp -r "{c}" "{code_dir}/"', shell=True)
            print(f"| Copied codes to {code_dir}.")
            trainer.fit(self)
        else:
            trainer.test(self)

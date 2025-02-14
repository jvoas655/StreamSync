import itertools
import logging
import os
import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import scipy
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler, Subset
from utils.utils import (fix_prefix, disable_print_if_not_master, get_obj_from_str,
                         instantiate_from_config, show_cfg_diffs)

logger = logging.getLogger(f'main.{__name__}')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_ddp(cfg):
    local_rank = os.environ.get('LOCAL_RANK')
    world_size = os.environ.get("WORLD_SIZE")
    if local_rank is not None:
        print(f'WORLDSIZE {os.environ.get("WORLD_SIZE")} - RANK {local_rank}')
        print('HOST:', os.environ.get('HOSTNAME'),
              'MASTER:', os.environ.get('MASTER_ADDR'), ':', os.environ.get('MASTER_PORT'))
        dist.init_process_group(cfg.training.dist_backend, world_size=int(world_size), timeout=timedelta(0, 600))
        cfg.training.local_rank = int(os.environ['LOCAL_RANK'])
        cfg.training.global_rank = int(os.environ['RANK'])
        cfg.training.world_size = dist.get_world_size()
        disable_print_if_not_master(is_master(dist.get_rank()))
        if is_master(dist.get_rank()):
            print(OmegaConf.to_yaml(cfg))
    else:
        cfg.training.local_rank = cfg.training.global_rank = 0
        print(OmegaConf.to_yaml(cfg))


def is_master(global_rank):
    return global_rank == 0


def get_curr_time_w_random_shift():
    # shifting for a random number of seconds so that exp folder names coincide less often
    now = datetime.now() - timedelta(seconds=np.random.randint(60))
    return now.strftime('%y-%m-%dT%H-%M-%S')


def broadcast_obj(object, global_rank, device):
    if dist.is_initialized():
        objects = [object if is_master(global_rank) else None for _ in range(dist.get_world_size())]
        dist.broadcast_object_list(objects, src=0, device=device)
        object = objects[0]
    return object


def get_device(cfg):
    device = torch.device(cfg.training.local_rank)
    torch.cuda.set_device(device)
    num_gpus = dist.get_world_size() if dist.is_initialized() else 1
    return device, num_gpus


def get_transforms(cfg):
    transforms = {}
    for mode in ['train', 'test']:
        ts_cfg = cfg.get(f'transform_sequence_{mode}', None)
        ts = [lambda x: x] if ts_cfg is None else [instantiate_from_config(c) for c in ts_cfg]
        transforms[mode] = torchvision.transforms.Compose(ts)
    return transforms


def get_datasets(cfg, transforms):
    DatasetClass = get_obj_from_str(cfg.data.dataset.target)
    load_fixed_offsets_on_test = cfg.data.dataset.params.load_fixed_offsets_on_test
    vis_load_backend = cfg.data.dataset.params.vis_load_backend
    size_ratio = cfg.data.dataset.params.size_ratio
    vids_path = cfg.data.vids_path
    data_cfg = cfg.data
    return {
        'train': DatasetClass('train', vids_path, transforms['train'],
                              load_fixed_offsets_on_test=load_fixed_offsets_on_test,
                              vis_load_backend=vis_load_backend, size_ratio=size_ratio, 
                              data_cfg=data_cfg),
        'valid': DatasetClass('valid', vids_path, transforms['test'],
                              load_fixed_offsets_on_test=load_fixed_offsets_on_test,
                              vis_load_backend=vis_load_backend, size_ratio=size_ratio, 
                              data_cfg=data_cfg),
        'valid-random': DatasetClass('valid-random', vids_path, transforms['test'],
                              load_fixed_offsets_on_test=load_fixed_offsets_on_test,
                              vis_load_backend=vis_load_backend, size_ratio=size_ratio, 
                              data_cfg=data_cfg),
        'test': DatasetClass('test', vids_path, transforms['test'],
                             load_fixed_offsets_on_test=load_fixed_offsets_on_test,
                             vis_load_backend=vis_load_backend, 
                              data_cfg=data_cfg),
    }


def get_batch_sizes(cfg, num_gpus):
    train_B = cfg.training.base_batch_size
    eval_B = cfg.training.base_batch_size_eval
    return {
        'train': train_B,
        'valid': eval_B,
        'test': eval_B,
    }


def get_loaders(cfg, datasets, batch_sizes):
    train_sampler = DistributedSampler(datasets['train'], shuffle=True) if dist.is_initialized() else None
    valid_sampler = DistributedSampler(datasets['valid'], shuffle=False) if dist.is_initialized() else None
    valid_random_sampler = DistributedSampler(datasets['valid'], shuffle=False) if dist.is_initialized() else None
    test_sampler = DistributedSampler(datasets['test'], shuffle=False) if dist.is_initialized() else None
    # what is the portion of the valid dataset to used for the `run_corrupted_valid` (turned off by default)
    subset_size = 0.2
    val_subset = Subset(datasets['valid'], torch.randperm(int(len(datasets['valid']) * subset_size)))
    valid_rand_aud_sampler = DistributedSampler(val_subset, shuffle=False) if dist.is_initialized() else None
    valid_rand_rgb_sampler = DistributedSampler(val_subset, shuffle=False) if dist.is_initialized() else None
    valid_perm_batch_sampler = DistributedSampler(val_subset, shuffle=False) if dist.is_initialized() else None
    return {
        'train': DataLoader(
            datasets['train'], batch_sizes['train'], shuffle=train_sampler is None,
            sampler=train_sampler, num_workers=cfg.training.num_workers, pin_memory=True),
        'valid': DataLoader(
            datasets['valid'], batch_sizes['valid'], shuffle=False,
            sampler=valid_sampler, num_workers=cfg.training.num_workers, pin_memory=True),
        'valid-random': DataLoader(
            datasets['valid-random'], batch_sizes['valid'], shuffle=False,
            sampler=valid_random_sampler, num_workers=cfg.training.num_workers, pin_memory=True),
        'test': DataLoader(
            datasets['test'], batch_sizes['test'], shuffle=False,
            sampler=test_sampler, num_workers=cfg.training.num_workers, pin_memory=True),
        'valid_rand_aud': DataLoader(
            val_subset, batch_sizes['valid'], shuffle=False,
            sampler=valid_rand_aud_sampler, num_workers=cfg.training.num_workers, pin_memory=True),
        'valid_rand_rgb': DataLoader(
            val_subset, batch_sizes['valid'], shuffle=False,
            sampler=valid_rand_rgb_sampler, num_workers=cfg.training.num_workers, pin_memory=True),
        'valid_perm_batch': DataLoader(
            val_subset, batch_sizes['valid'], shuffle=False,
            sampler=valid_perm_batch_sampler, num_workers=cfg.training.num_workers, pin_memory=True),
    }


def get_model(cfg, device):
    model = instantiate_from_config(cfg.model)

    # TODO: maybe in the module
    if cfg.model.params.vfeat_extractor.is_trainable is False:
        for params in model.vfeat_extractor.parameters():
            params.requires_grad = False
    if cfg.model.params.afeat_extractor.is_trainable is False:
        for params in model.afeat_extractor.parameters():
            params.requires_grad = False

    model = model.to(device)
    model_without_ddp = model
    if dist.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
        # any mistaken calls on `model_without_ddp` (=None) will likely raise an error
        model_without_ddp = model.module
    else:
        model = torch.nn.DataParallel(model)

    return model, model_without_ddp


def get_optimizer(cfg, model, num_gpus):
    learning_rate = cfg.training.base_learning_rate * num_gpus
    # TODO: instantiate (but we need to pass params as well - fix the intantiate fn)
    # TODO: consider uncommenting this line and optimize only these tensors (from torchvision reference)
    # params = [p for p in model.parameters() if p.requires_grad]
    # avoiding NaN during half precision training
    eps = 1e-7 if cfg.training.use_half_precision else 1e-8
    if cfg.training.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, cfg.training.optimizer.betas,
                                     eps, cfg.training.optimizer.weight_decay)
    elif cfg.training.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate, cfg.training.optimizer.betas,
                                      eps, cfg.training.optimizer.weight_decay)
    elif cfg.training.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, cfg.training.optimizer.momentum,
                                    weight_decay=cfg.training.optimizer.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer: "{cfg.training.optimizer.name}" is not implemented')
    return optimizer


def get_lr_scheduler(cfg, optimizer):
    if cfg.training.lr_scheduler.name == 'constant_with_warmup':
        assert 'warmup' in cfg.training.lr_scheduler, f'{cfg.training.lr_scheduler}'
        warmup = cfg.training.lr_scheduler.warmup
        lr_sched = lr_scheduler.SequentialLR(optimizer, schedulers=[
            lr_scheduler.LinearLR(optimizer, start_factor=1/100, total_iters=warmup),
            lr_scheduler.ConstantLR(optimizer, factor=1),
        ], milestones=[warmup])
    elif cfg.training.lr_scheduler.name == 'constant':
        lr_sched = lr_scheduler.ConstantLR(optimizer, factor=1)
    elif cfg.training.lr_scheduler.name == 'warmup_constant_decay':
        warmup = cfg.training.lr_scheduler.warmup# * 860 # steps per epoch
        constant = cfg.training.lr_scheduler.constant# * 860
        remaining = cfg.training.lr_scheduler.remaining# * 860
        lr_sched = lr_scheduler.SequentialLR(optimizer, schedulers=[
            lr_scheduler.LinearLR(optimizer, start_factor=1/100, total_iters=warmup),
            lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=constant),
            lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1/10000, total_iters=remaining)
        ], milestones=[warmup, warmup+constant])
    return lr_sched


# Add resume option
def load_ckpt(cfg, model_wo_ddp, optimizer, scaler, lr_scheduler):
    ckpt = torch.load(cfg.ckpt_path, map_location=torch.device('cpu'))
    ckpt_cfg = ckpt['args']
    try:
        model_wo_ddp.load_state_dict(ckpt['model'])
        model_only_params = [] # Perfect match
    except:
        d1 = ckpt['model']
        d2 = model_wo_ddp.state_dict()
        ckpt_only_params = []
        model_only_params = []
        for k in d1:
            if k not in d2:
                ckpt_only_params.append(k)
        for k in d2:
            if k not in d1:
                model_only_params.append(k)
        assert(len(ckpt_only_params) == 0)
        model_wo_ddp.load_state_dict(ckpt['model'], strict=False)
    if cfg.resume_scheduler_opt_scaler:
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    start_epoch = ckpt['epoch']
    # restarting training counters if the ckpt is used to init weights rather than continuing training
    if cfg.training.finetune:
        old_vid_dir = f'{Path(ckpt_cfg.data.vids_path).stem}/{Path(ckpt_cfg.data.vids_path).parent.stem}'
        new_vid_dir = f'{Path(cfg.data.vids_path).stem}/{Path(cfg.data.vids_path).parent.stem}'
        # assert old_vid_dir != new_vid_dir, f'old: {old_vid_dir}; new: {new_vid_dir}'
        # assert ckpt_cfg.data.dataset.target != cfg.data.dataset.target, ckpt_cfg.data.dataset.target
        logger.info(f'Finetuning from: {ckpt_cfg.ckpt_path} on {cfg.data.dataset.target}')
        ckpt['metrics'][cfg.training.metric_name] = 0 if cfg.training.to_max_metric else float('inf')
        start_epoch = 0
        show_cfg_diffs(ckpt_cfg, cfg, Path(cfg.ckpt_path).parent / 'cfg_diffs.diff')
    elif cfg.training.resume:
        start_epoch += 1
    # a bit ugly but it patches checkpoints produced by the 'old' code
    metrics = ckpt['metrics']
    metrics = metrics.get('off', metrics)
    return start_epoch, metrics, model_only_params


class EarlyStopper(object):

    def __init__(self, patience: int, to_max: bool, metric_name: str) -> None:
        '''E.g. If loss is the trackable metric, to_max=False; if accuracy, to_max=True'''
        self.no_change_epochs = 0
        self.to_max = to_max
        self.best_metric = 0.0 if to_max else float('inf')
        self.metric_name = metric_name
        self.triggered = False
        self.patience = patience

    def decide(self, global_rank, logger, metrics):
        new_metric = metrics[self.metric_name]

        if is_master(global_rank):
            logger.print_logger.info(f'decide inputs: {new_metric:.6f}; best: {self.best_metric}')

        is_new_model_better_than_curr = self.is_new_model_better_than_curr(new_metric)

        if is_new_model_better_than_curr:
            self.no_change_epochs = 0
            self.best_metric = new_metric
        else:
            self.no_change_epochs += 1
            if is_master(global_rank):
                logger.print_logger.info(
                    f'{self.metric_name} ({self.best_metric:.5f}) hasnt changed for {self.no_change_epochs} '
                    f'patience: {self.patience}'
                )
            if self.no_change_epochs >= self.patience:
                self.triggered = True

        return is_new_model_better_than_curr

    def is_new_model_better_than_curr(self, new_metric_val: float):
        return self.best_metric < new_metric_val if self.to_max else self.best_metric > new_metric_val


def toggle_mode(cfg, model, phase, epoch, model_only_params):
    mode = -1
    if phase == 'train':
        model.train()
        if cfg.training.freeze_first > epoch: # Only freeze for the first few epochs
            for n,p in model.named_parameters():
                if (n in model_only_params) or (n[7:] in model_only_params): # First 7 chars are 'module.' if dist.is_initialized()
                    p.requires_grad = True # Parameters not loaded from the ckpt should be trained
                else:
                    p.requires_grad = False # Parameters loaded from the ckpt are frozen
            mode = 1
        elif cfg.training.freeze_first <= epoch:
            # First epoch back, so need to un-freeze everything
            for n,p in model.named_parameters():
                p.requires_grad = True
            # Re-freeze extractors if not trainable
            if cfg.model.params.vfeat_extractor.is_trainable is False:
                if dist.is_initialized():
                    for params in model.module.vfeat_extractor.parameters():
                        params.requires_grad = False
                else:
                    for params in model.vfeat_extractor.parameters():
                        params.requires_grad = False
            if cfg.model.params.afeat_extractor.is_trainable is False:
                if dist.is_initialized():
                    for params in model.module.afeat_extractor.parameters():
                        params.requires_grad = False
                else:
                    for params in model.afeat_extractor.parameters():
                        params.requires_grad = False
            mode = 2
        if cfg.model.params.afeat_extractor.is_trainable is False:
            if dist.is_initialized():
                model.module.afeat_extractor.eval()
            else:
                model.afeat_extractor.eval()
        if cfg.model.params.vfeat_extractor.is_trainable is False:
            if dist.is_initialized():
                model.module.vfeat_extractor.eval()
            else:
                model.vfeat_extractor.eval()
    else:
        model.eval()
    return mode


def prepare_inputs(batch, device, phase=None):
    targets = batch['targets']
    targets = {k: targets[k].to(device) for k in ['vggsound_target', 'offset_target'] if k in targets}

    aud = batch['audio'].to(device)
    vid = batch['video'].to(device)

    # if `training.run_corrupted_val`
    if phase == 'valid_rand_aud':
        aud = (aud.max() - aud.min()) * torch.rand(aud.shape, dtype=aud.dtype, device=aud.device) + aud.min()
    elif phase == 'valid_rand_rgb':
        vid = (vid.max() - vid.min()) * torch.rand(vid.shape, dtype=vid.dtype, device=vid.device) + vid.min()
    elif phase == 'valid_perm_batch':
        B = vid.shape[0]
        perm_idx = torch.randperm(B)
        vid = vid[perm_idx, ...]

    return aud, vid, targets


def apply_batch_mixup(x, alpha):
    if alpha > 0.0:
        B = x.shape[0]
        perm_idx = torch.randperm(B)
        # lmbd = torch.distributions.Beta(alpha, alpha).sample()  # this is not what we need
        lmbd = torch.distributions.Uniform(low=0.0, high=alpha).sample()
        return lmbd * x + (1 - lmbd) * x[perm_idx, ...]
    else:
        return x

def make_backward_step(cfg, loss, model, scaler, retain_graph=False):
    # without half precision training:
    # loss.backward()
    # if cfg.get('max_clip_norm', None) is not None:
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_clip_norm)
    # optimizer.step()
    scaler.scale(loss).backward(retain_graph=retain_graph)

def make_optim_step(cfg, model, optimizer, scaler, lr_scheduler):
    # without half precision training:
    # loss.backward()
    # if cfg.get('max_clip_norm', None) is not None:
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_clip_norm)
    # optimizer.step()
    scaler.unscale_(optimizer)
    max_clip_norm = cfg.training.get('max_clip_norm', None)
    if max_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    lr_scheduler.step()


def make_backward_and_optim_step(cfg, loss, model, optimizer, scaler, lr_scheduler):
    # without half precision training:
    # loss.backward()
    # if cfg.get('max_clip_norm', None) is not None:
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_clip_norm)
    # optimizer.step()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    max_clip_norm = cfg.training.get('max_clip_norm', None)
    if max_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    lr_scheduler.step()


def verbose_iter_progress(logger, prog_bar, iter_step, iter_results, phase):
    # iter logging (making it a bit more sparse for faster tboard loading)
    if iter_step % 100 == 0:
        iter_loss = iter_results['loss_total']
        if phase not in ['valid_rand_aud', 'valid_rand_rgb', 'valid_perm_batch']:
            logger.log_iter_loss(iter_loss, iter_step, phase, prefix='total')
        # tracks loss in the tqdm progress bar
        prog_bar.set_postfix(loss=iter_results['loss_total'])


def verbose_log(logger, prog_bar, iter_step, lr, loss):
    if (iter_step % 4 == 0):
        logger.add_scalar('lr', lr, iter_step)
        logger.add_scalar('loss', loss, iter_step)
    prog_bar.set_postfix(lr=lr, loss=loss)


def verbose_epoch_progress(global_rank, logger, running_results, phase, epoch):
    running_results = gather_dict(running_results)
    for key, value in running_results.items():
        print(key, type(value))
    # logging loss values
    if is_master(global_rank):
        logger.log_epoch_loss(running_results['loss_total'], epoch, phase, prefix='total')
    # logging metrics
    for key, value in running_results['logits'].items():
        running_results['logits'][key] = torch.cat(value).float()
    targets = torch.cat(running_results['targets']).float()
    metrics = calc_metrics(targets, running_results['logits'])
    if is_master(global_rank):
        logger.log_epoch_metrics(metrics, epoch, phase)
    epoch_loss = running_results['loss_total']
    return epoch_loss, metrics


def verbose_test_progress(global_rank, logger, cfg, running_results, ckpt_epoch):
    running_results = gather_dict(running_results)

    if is_master(global_rank):
        logits = torch.cat(running_results['logits']).float()
        targets = torch.cat(running_results['targets']).float()
        metrics = calc_metrics(targets, logits)
        metrics['loss'] = running_results['loss_total']
        logger.log_test_metrics(metrics, dict(cfg), ckpt_epoch)
        logger.print_logger.info('Finished the experiment')


def gather_dict(dct):
    if dist.is_initialized():
        dist.barrier()
        for k, v in dct.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    gather_buffer = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(gather_buffer, sub_v)
                    dct[k][sub_k] = list(itertools.chain(*gather_buffer))
            else:
                gather_buffer = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gather_buffer, v)
                if isinstance(v, list):
                    # flattens a list of lists into one list
                    dct[k] = list(itertools.chain(*gather_buffer))
                elif isinstance(v, float):
                    # average
                    dct[k] = sum(gather_buffer) / len(gather_buffer)
                else:
                    raise NotImplementedError(f'{type(v)}, \n{v}')
    return dct


def calc_metrics(targets: torch.Tensor, outputs_dict: Dict[str, torch.FloatTensor], topk=(1, 5), only_accuracy=False, prefix=''):
    """
    Adapted from https://github.com/hche11/VGGSound/blob/master/utils.py

    Calculate statistics including mAP, AUC, and d-prime.
        Args:
            output: 2d tensors, (dataset_size, classes_num) - before softmax
            target: 1d tensors, (dataset_size, )
            topk: tuple
        Returns:
            metric_dict: a dict of metrics
    """
    prefix = fix_prefix(prefix)
    metrics_dict = dict()
    targets_store = targets
    for key, value in outputs_dict.items():
        targets = targets_store
        outputs = value
        dataset_size, num_cls = outputs.shape
        topk = [min(k, num_cls) for k in topk]

        if np.isinf(outputs.numpy()).any() or not np.isfinite(outputs.numpy()).all():
            outputs = torch.rand_like(outputs)
            logger.warning('infinity or loss was nan. Replacing with random')

        _, preds = torch.topk(outputs, k=max(topk), dim=1)

        # accuracy@k
        # print('targets', targets)
        target_idxs = torch.max(targets, axis=-1).indices
        # print('target idxs', target_idxs)
        targets_for_acc = target_idxs.view(-1, 1).expand_as(preds)
        # print('for acc', targets_for_acc)
        correct_for_maxtopk = preds == targets_for_acc
        for k in topk:
            TPs = correct_for_maxtopk[:, :k].sum()
            topk_accuracy = float(TPs / dataset_size)
            metrics_dict[f'{prefix}accuracy_{k}_{key}'] = topk_accuracy

        # accuracy@k_tol
        if num_cls == 3 and dataset_size > 100:  # 100 is to avoid verbosity on iteration level
            logger.warning('Accuracy with tolerance is not reliable as num of offset classes is 3.')
        targets_for_acc_left_tol = (targets_for_acc - 1).clamp(0, num_cls-1)
        targets_for_acc_right_tol = (targets_for_acc + 1).clamp(0, num_cls-1)
        targets_for_acc_w_tol = torch.stack([targets_for_acc_left_tol, targets_for_acc, targets_for_acc_right_tol])
        correct_for_maxtopk_w_tol = (preds == targets_for_acc_w_tol).any(dim=0)
        for k in topk:
            # tolerance might result in having more than one `True` per item. Preventing overcounting w/ `any()`
            TPs_w_tol = correct_for_maxtopk_w_tol[:, :k].any(dim=1).sum()
            topk_accuracy_w_tol = float(TPs_w_tol / dataset_size)
            metrics_dict[f'{prefix}accuracy_{k}_tol1_{key}'] = topk_accuracy_w_tol


        # avg precision, average roc_auc, and dprime
        unique_targets = sorted(list(set(target_idxs.tolist())))
        targets = torch.nn.functional.one_hot(target_idxs, num_classes=num_cls)

        # ids of the predicted classes (same as softmax)
        targets_pred = torch.softmax(outputs, dim=1)

        targets = targets.numpy()
        targets_pred = targets_pred.numpy()

        # one-vs-rest
        avg_p = [average_precision_score(targets[:, c], targets_pred[:, c], average=None) for c in range(num_cls)]
        try:
            roc_aucs = [roc_auc_score(targets[:, c], targets_pred[:, c], average=None) for c in range(num_cls)]
        except ValueError:
            logger.warning('Weird... Some classes never occured in targets. Do not trust the metrics.')
            # logger.warning(f'Here is the list of {prefix} target classes: {unique_targets}')
            roc_aucs = np.array([0.5])
            avg_p = np.array([0])

        metrics_dict[f'{prefix}mAP_{key}'] = np.mean(avg_p)
        metrics_dict[f'{prefix}mROCAUC_{key}'] = np.mean(roc_aucs)
        # Percent point function (ppf) (inverse of cdf — percentiles).
        metrics_dict[f'{prefix}dprime_{key}'] = scipy.stats.norm().ppf(metrics_dict[f'{prefix}mROCAUC_{key}'])*np.sqrt(2)

    return metrics_dict


if __name__ == '__main__':
    targets = torch.tensor([3, 3, 1, 2, 1, 0])
    outputs = torch.tensor([
        [.2, .3, .1, .5],
        [.4, .3, .0, .1],
        [.5, .1, .4, .3],
        [.0, .2, .4, .5],
        [.2, .3, .1, .1],
        [.2, .1, .1, .1],
    ]).float()
    metrics_dict = calc_metrics(targets, outputs, topk=(1, 3))
    print(metrics_dict)

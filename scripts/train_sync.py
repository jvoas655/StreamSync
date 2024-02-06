import random

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import pickle as pkl
from utils.logger import LoggerWithTBoard

from scripts.train_utils import (EarlyStopper, apply_batch_mixup,
                                 broadcast_obj, get_batch_sizes, get_datasets,
                                 get_device, get_loaders, get_lr_scheduler,
                                 get_model, get_optimizer, get_transforms,
                                 init_ddp, is_master, load_ckpt,
                                 make_backward_and_optim_step, 
                                 make_backward_step, make_optim_step, prepare_inputs,
                                 set_seed, toggle_mode, verbose_epoch_progress,
                                 verbose_iter_progress, verbose_log,
                                 verbose_test_progress)


def run_validation(cfg, global_rank, logger, early_stopper, scaler, optimizer, lr_scheduler, epoch, iter_step, device, streaming, num_streaming_iters, model, model_without_ddp, model_only_params, loaders):
    phase = 'valid'
    # does model.eval() or .train() on appropriate submodules
    toggle_mode(cfg, model, phase, epoch, model_only_params)

    # init runnining results
    running_results = dict(logits = {}, targets=[], loss_total=0)

    if dist.is_initialized():
        loaders[phase].sampler.set_epoch(epoch)


    prog_bar = tqdm(loaders[phase], f'{phase} ({epoch} - {iter_step})', ncols=0)
    for i, batch in enumerate(prog_bar):
        # sends targets and inputs to cuda
        aud, vid, targets = prepare_inputs(batch, device, phase)
        
        
        if (streaming):
            streaming_features = None
            acum_loss = 0.0
            iter_results = dict(logits = {})
            # gradient and half-precision toggles
            with torch.no_grad():
                with torch.autocast("cuda", enabled=cfg.training.use_half_precision):
                    for streaming_iter in range(num_streaming_iters):
                        if (hasattr(model_without_ddp, "streaming_head")):
                            streaming_loss, singleshot_loss, streaming_logits, ss_logits, streaming_features = model(
                                vid[:, streaming_iter, ...], 
                                aud[:, streaming_iter, ...], 
                                streaming_features, 
                                targets
                            )
                            acum_loss += 0.5 * (streaming_loss.mean() + singleshot_loss.mean()) / num_streaming_iters
                            iter_results['logits'][f'streaming_logits_iter{streaming_iter}'] = [streaming_logits.detach().cpu()]
                            iter_results['logits'][f'singleshot_logits_iter{streaming_iter}'] = [ss_logits.detach().cpu()]
                        else:
                            loss, logits = model(
                                vid[:, streaming_iter, ...], 
                                aud[:, streaming_iter, ...], 
                                targets
                            )
                            acum_loss += loss.mean() / num_streaming_iters
                            iter_results['logits'][f'singleshot_logits_iter{streaming_iter}'] = [logits.detach().cpu()]

            # gathering results in one place to iterate on this later
            iter_results['loss_total'] = acum_loss.item()
            iter_results['targets'] = [targets['offset_target'].cpu()]
        
        else:
            acum_loss = 0.0
            iter_results = dict(logits = {})
            # gradient and half-precision toggles
            with torch.no_grad():
                with torch.autocast("cuda", enabled=cfg.training.use_half_precision):
                    
                        
                    loss, logits = model(
                        vid, 
                        aud, 
                        targets
                    )
                    acum_loss += loss.mean()
                    iter_results['logits'][f'singleshot_logits'] = [logits.detach().cpu()]

            # gathering results in one place to iterate on this later
            iter_results['loss_total'] = acum_loss.item()
            iter_results['targets'] = [targets['offset_target'].cpu()]

        # doing it here instead of the dict() because we would like to verbose unscaled loss values
        iter_results[f'loss_total'] /= len(loaders[phase])

        # update running results
        for k in running_results.keys():
            if (isinstance(running_results[k], dict)):
                for sub_key in iter_results[k].keys():
                    if (sub_key not in running_results[k]):
                        running_results[k][sub_key] = []
                    running_results[k][sub_key] += iter_results[k][sub_key]
            else:
                running_results[k] += iter_results[k]
    # logs epoch metrics to tensorboard/wandb
    epoch_loss, metrics = verbose_epoch_progress(global_rank, logger, running_results, phase, iter_step)

    # Early stopping update
    has_model_improved = early_stopper.decide(global_rank, logger, metrics)
    if has_model_improved and is_master(global_rank):
        # saves the best checkpoint. Replaces the previous one
        logger.log_best_model(model_without_ddp, scaler, epoch_loss,
                                iter_step, optimizer, lr_scheduler, metrics, cfg)

    # wait for other workers to get here
    if dist.is_initialized():
        dist.barrier()

    if early_stopper.triggered:
        if is_master(global_rank):
            logger.print_logger.info(f'Training is early stopped @ {iter_step}; RANK: {global_rank}')
        return True
    else:
        return False

def train(cfg):
    init_ddp(cfg)
    global_rank = dist.get_rank() if dist.is_initialized() else cfg.training.global_rank
    # LoggerWithTBoard inherits Tensorboard summary module and, therefore, can be treated as one on steroids
    logger = LoggerWithTBoard(global_rank, cfg)

    set_seed(cfg.training.seed + global_rank)

    # makes iterations faster if your inputs are of a fixed size
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    torch.backends.cudnn.benchmark = True

    device, num_gpus = get_device(cfg)

    # ckpt_path was created only for the master (to keep it the same), now we broadcast it to each worker
    cfg.ckpt_path = broadcast_obj(cfg.ckpt_path, global_rank, device)
    # making sure each worker has the same ckpt path as the master
    #print("\n------", device, num_gpus, "-----\n")
    assert hasattr(cfg, 'ckpt_path'), f'I AM AT RANK: {global_rank}'

    transforms = get_transforms(cfg)
    datasets = get_datasets(cfg, transforms)
    batch_sizes = get_batch_sizes(cfg, num_gpus)
    loaders = get_loaders(cfg, datasets, batch_sizes)
    model, model_without_ddp = get_model(cfg, device)
    
    optimizer = get_optimizer(cfg, model, num_gpus)
    lr_scheduler = get_lr_scheduler(cfg, optimizer)

    streaming = hasattr(cfg.data, "streaming") and cfg.data.streaming.enabled

    if (hasattr(cfg.data, "streaming")):
        num_streaming_iters = cfg.data.streaming.max_streaming_frames
    else:
        num_streaming_iters = None

    accumulation_steps = cfg.training.get('accumulation_steps', 1)

    logger.log_param_num(global_rank, model)

    # Add DataParallel for multiple GPUS
    #model = torch.nn.DataParallel(model)
    model_only_params = [] # Written over in loading checkpoint; used to freeze loaded params while training new ones if freeze_first > 0

    early_stopper = EarlyStopper(cfg.training.patience, cfg.training.to_max_metric, cfg.training.metric_name)

    # the scaller for the loss. Helps to avoid precision underflow during half prec training
    scaler = torch.cuda.amp.GradScaler()

    # this chunk has a complicate logic but it simply loads pre-trained ckpt during finetuning/resuming
    if cfg.training.run_test_only or cfg.training.resume or cfg.training.finetune:
        start_epoch, metrics, model_only_params = load_ckpt(cfg, model_without_ddp, optimizer, scaler, lr_scheduler)
        if (cfg.training.metric_name in metrics):
            early_stopper.best_metric = metrics[cfg.training.metric_name]
    else:
        start_epoch = 0

    # don't do training loops if a user wants to only probe the model on the test set
    num_epochs = 0 if cfg.training.run_test_only else cfg.training.num_epochs
    # loop over the train and validation multiple times (typical PT boilerplate)
    for epoch in range(start_epoch, num_epochs):

        #if 'run_corrupted_val' not in cfg.training or cfg.training.run_corrupted_val:
        #    phases_to_run_on.extend(['valid_rand_aud', 'valid_rand_rgb', 'valid_perm_batch'])
        phase = 'train'
        # does model.eval() or .train() on appropriate submodules
        iter_step = epoch * len(loaders[phase])
        if (iter_step == 0):
            if dist.is_initialized():
                dist.barrier()
            #run_validation(cfg, global_rank, logger, early_stopper, scaler, optimizer, lr_scheduler, epoch, iter_step, device, True, num_streaming_iters, model, model_without_ddp, model_only_params, loaders)
            torch.cuda.empty_cache()
        # Mode 1 means training pretrained weights frozen
        # Mode 2 mean training with all weights unfrozen
        mode = toggle_mode(cfg, model, phase, iter_step, model_only_params)
        params = 0
        trainable = 0
        for param in model.parameters():
            params += torch.numel(param)
            if (param.requires_grad):
                trainable += torch.numel(param)
        print(f"{trainable // 1024}K / {params // 1024}K")
        # init runnining results
        running_results = dict(logits = {}, targets=[], loss_total=0)
        running_loss = []

        if dist.is_initialized():
            loaders[phase].sampler.set_epoch(epoch)


        prog_bar = tqdm(loaders[phase], f'{phase} ({epoch})', ncols=0)
        for i, batch in enumerate(prog_bar):

            iter_step = epoch * len(loaders[phase]) + i
            # zero the parameter gradients
            if iter_step % accumulation_steps == 0:
                optimizer.zero_grad()

            # sends targets and inputs to cuda
            aud, vid, targets = prepare_inputs(batch, device, phase)

            aud = apply_batch_mixup(aud, cfg.training.mixup_alpha)
            
            if (streaming):
                streaming_features = None
                acum_loss = 0.0
                iter_results = dict(logits = {})
                back_prop_cutoff_iter = iter_step % (num_streaming_iters - cfg.data.streaming.num_frames_to_train_on * 2) + cfg.data.streaming.num_frames_to_train_on #random.randint(0, num_streaming_iters - cfg.data.streaming.num_frames_to_train_on) if cfg.data.streaming.num_frames_to_train_on < num_streaming_iters else 0
                '''
                for streaming_iter in range(num_streaming_iters):
                    # gradient and half-precision toggles
                    if mode == 2 and (streaming_iter < back_prop_cutoff_iter):
                        with torch.set_grad_enabled(False):
                            with torch.autocast("cuda", enabled=cfg.training.use_half_precision):
                                streaming_loss, singleshot_loss, streaming_logits, ss_logits, streaming_features = model(
                                    vid[:, streaming_iter, ...], 
                                    aud[:, streaming_iter, ...], 
                                    streaming_features, 
                                    targets
                                )
                    elif mode == 2 and (streaming_iter >= back_prop_cutoff_iter + cfg.data.streaming.num_frames_to_train_on):
                          break # Break to save iterations of inference
                    else:
                        with torch.set_grad_enabled(True):
                            with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
                                
                                # saves recontructed input to the model during the first iteration (detects bugs)
                                #if is_master(global_rank) and iter_step == 0 and phase in ['train', 'valid']:
                                #    logger.vizualize_input(vid[:, streaming_iter, ...], aud[:, streaming_iter, ...], batch, iter_step)
                                streaming_loss, singleshot_loss, streaming_logits, ss_logits, streaming_features = model(
                                    vid[:, streaming_iter, ...], 
                                    aud[:, streaming_iter, ...], 
                                    streaming_features, 
                                    targets
                                )
                                acum_loss += 0.5 * (streaming_loss.mean() + singleshot_loss.mean()) / cfg.data.streaming.num_frames_to_train_on
                    iter_results['logits'][f'streaming_logits_iter{streaming_iter}'] = [streaming_logits.detach().cpu()]
                    iter_results['logits'][f'singleshot_logits_iter{streaming_iter}'] = [ss_logits.detach().cpu()]
                '''
                loss_norm_factor = 0.0
                for streaming_iter in range(num_streaming_iters):
                    # gradient and half-precision toggles
                    
                    with torch.set_grad_enabled(True):
                        with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
                            
                            # saves recontructed input to the model during the first iteration (detects bugs)
                            #if is_master(global_rank) and iter_step == 0 and phase in ['train', 'valid']:
                            #    logger.vizualize_input(vid[:, streaming_iter, ...], aud[:, streaming_iter, ...], batch, iter_step)
                            streaming_loss, singleshot_loss, streaming_logits, ss_logits, streaming_features = model(
                                vid[:, streaming_iter, ...], 
                                aud[:, streaming_iter, ...], 
                                streaming_features, 
                                targets,
                                detach_base=(mode == 2 and (streaming_iter < back_prop_cutoff_iter or streaming_iter >= back_prop_cutoff_iter + cfg.data.streaming.num_frames_to_train_on))
                            )
                            if (streaming_iter < cfg.data.streaming.num_frames_to_train_on - 1):
                                acum_loss += ((1/3) * singleshot_loss.mean())
                                loss_norm_factor += (1/3)
                            else:
                                acum_loss += ((1/3) * singleshot_loss.mean() + (2/3) * streaming_loss.mean())
                                loss_norm_factor += 1
                    iter_results['logits'][f'streaming_logits_iter{streaming_iter}'] = [streaming_logits.detach().cpu()]
                    iter_results['logits'][f'singleshot_logits_iter{streaming_iter}'] = [ss_logits.detach().cpu()]
                acum_loss /= loss_norm_factor

                # gathering results in one place to iterate on this later
                iter_results['loss_total'] = acum_loss.item()
                iter_results['targets'] = [targets['offset_target'].cpu()]
                running_loss.append(iter_results['loss_total'])
                if (len(running_loss) > 10):
                    running_loss = running_loss[-10:]
                if (accumulation_steps > 1):
                    make_backward_step(cfg, acum_loss / accumulation_steps, model, scaler)
                    if (iter_step + 1) % accumulation_steps == 0:
                        make_optim_step(cfg, model, optimizer, scaler, lr_scheduler)
                else:
                    make_backward_and_optim_step(cfg, acum_loss, model, optimizer, scaler, lr_scheduler)
                
            else:
                acum_loss = 0.0
                iter_results = dict(logits = {})
                # half-precision toggles
                with torch.autocast("cuda", enabled=cfg.training.use_half_precision):
                    
                        
                    loss, logits = model(
                        vid, 
                        aud, 
                        targets
                    )
                    acum_loss += loss.mean()
                    iter_results['logits'][f'singleshot_logits'] = [logits.detach().cpu()]

                # gathering results in one place to iterate on this later
                iter_results['loss_total'] = acum_loss.item()
                iter_results['targets'] = [targets['offset_target'].cpu()]
                running_loss.append(iter_results['loss_total'])
                if (len(running_loss) > 10):
                    running_loss = running_loss[-10:]
                
                if (accumulation_steps > 1):
                    make_backward_step(cfg, acum_loss / accumulation_steps, model, scaler)
                    if (iter_step + 1) % accumulation_steps == 0:
                        make_optim_step(cfg, model, optimizer, scaler, lr_scheduler)
                else:
                    make_backward_and_optim_step(cfg, acum_loss, model, optimizer, scaler, lr_scheduler)
                
                
            if is_master(global_rank):
                #verbose_iter_progress(logger, prog_bar, iter_step, iter_results, phase)
                verbose_log(logger, prog_bar, iter_step, lr_scheduler.get_last_lr()[0], np.mean(running_loss))

            # doing it here instead of the dict() because we would like to verbose unscaled loss values
            iter_results[f'loss_total'] /= len(loaders[phase])

            # update running results
            for k in running_results.keys():
                if (isinstance(running_results[k], dict)):
                    for sub_key in iter_results[k].keys():
                        if (sub_key not in running_results[k]):
                            running_results[k][sub_key] = []
                        running_results[k][sub_key] += iter_results[k][sub_key]
                else:
                    running_results[k] += iter_results[k]
            if (iter_step > 0 and iter_step % cfg.training.val_per_num_iters  == 0):
                if dist.is_initialized():
                    dist.barrier()
                if (not streaming):
                    verbose_epoch_progress(global_rank, logger, running_results, phase, iter_step)
                running_results = dict(logits = {}, targets=[], loss_total=0)
                if dist.is_initialized():
                    dist.barrier()
                run_validation(cfg, global_rank, logger, early_stopper, scaler, optimizer, lr_scheduler, epoch, iter_step, device, True, num_streaming_iters, model, model_without_ddp, model_only_params, loaders)
                torch.cuda.empty_cache()
                mode = toggle_mode(cfg, model, phase, iter_step, model_only_params)
                params = 0
                trainable = 0
                for param in model.parameters():
                    params += torch.numel(param)
                    if (param.requires_grad):
                        trainable += torch.numel(param)
                print(f"{trainable // 1024}K / {params // 1024}K")


    if is_master(global_rank):
        logger.print_logger.info('Finished Training')

    # Testing the model
    phase = 'test'
    cfg.training.finetune = False
    # loading the best model
    ckpt_epoch, best_metric_val, model_only_params = load_ckpt(cfg, model_without_ddp, optimizer, scaler, lr_scheduler)
    if is_master(global_rank):
        logger.print_logger.info(f'Loading the best model from {cfg.ckpt_path}')
        logger.print_logger.info(f'Best metric: {best_metric_val}')
        logger.print_logger.info((f'The model was trained for {ckpt_epoch} epochs.'))
    model.eval()

    # init runnining results
    running_results = dict(logits=[], targets=[], loss_total=0)

    if dist.is_initialized():
        loaders[phase].sampler.set_epoch(ckpt_epoch)

    prog_bar = tqdm(loaders[phase], f'{phase} ({ckpt_epoch})', ncols=0)
    all_dumped_attns = []
    for iter_step, batch in enumerate(prog_bar):
        # sends inputs and targets to cuda
        aud, vid, targets = prepare_inputs(batch, device, phase)
        # zero the parameter gradients
        optimizer.zero_grad()
        # gradient and half-precision toggles
        with torch.set_grad_enabled(False):
            with torch.autocast("cuda", enabled=cfg.training.use_half_precision):
                if cfg.training.dump_attn_weights and len(all_dumped_attns) < 100:
                    new_attn_dict = {
                        'path': batch['path'],
                        'targets': batch['targets'],
                        'start': batch['start'],
                    }
                    if cfg.model.params.transformer.params.ablate_mixer:
                        # Only one round of selectors
                        loss, logits, vsa1, asa1, vca1, aca1 = model(vid, aud, targets, return_attn_weights=True)
                    else:
                        loss, logits, vsa1, asa1, vca1, aca1, vsa2, asa2, vca2, aca2 = model(vid, aud, targets, return_attn_weights=True)
                        new_attn_dict['vis_self_attn_2'] = vsa2
                        new_attn_dict['aud_self_attn_2'] = asa2
                        new_attn_dict['vis_cross_attn_2'] = vca2
                        new_attn_dict['aud_cross_attn_2'] = aca2
                    new_attn_dict['loss'] = loss.detach().cpu()
                    new_attn_dict['logits'] = logits.detach().cpu()
                    new_attn_dict['vis_self_attn_1'] = vsa1
                    new_attn_dict['aud_self_attn_1'] = asa1
                    new_attn_dict['vis_cross_attn_1'] = vca1
                    new_attn_dict['aud_cross_attn_1'] = aca1
                    for k in new_attn_dict.keys():
                        if 'attn' in k:
                            for l in range(len(new_attn_dict[k])):
                                new_attn_dict[k][l].detach().cpu()
                    all_dumped_attns.append(new_attn_dict)
                else:
                    loss, logits = model(vid, aud, targets)

        pkl.dump(all_dumped_attns, open('dumped_attention_weights.pkl','wb'))
        # gathering results in one place to iterate on this later
        try:
            iter_results = dict(
                logits=[logits.detach().cpu()],
                targets=[targets['offset_target'].cpu()],
                loss_total=loss.mean().item() / len(loaders[phase]),
            )
        except:
            print('Failed!')
            print('loss is', loss)
            print('with .mean().item() we get', loss.mean().item())
            print('loaders[phase] is', loaders[phase])
            exit(0)
        for k in running_results.keys():
            running_results[k] += iter_results[k]


    # logs test metrics to tensorboard/wandb
    verbose_test_progress(global_rank, logger, cfg, running_results, ckpt_epoch)

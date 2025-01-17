# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer

import utils

from torch.nn import functional as F

def replace_with_ncm(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):

    model.eval()
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        feature_list = []
        with torch.no_grad():
            if args.multi_query==False:
                if original_model is not None:
                    output = original_model(input)
                    cls_features = output['pre_logits']
                    
                else:
                    cls_features = None
            else:
                for i in range(task_id):
                    feature_list.append(torch.zeros((input.shape[0], 768)).to(device))
                output = model(input, task_id=task_id,query=True)
                feature_list.append(output['pre_logits'])
                for i in range(task_id+1, args.num_tasks):
                    feature_list.append(torch.zeros((input.shape[0], 768)).to(device))
                cls_features = torch.stack(feature_list, dim=1)
                # print(cls_features.shape)
        
            output = model(input, task_id=task_id, cls_features=cls_features, train=True,target=target)

        # print('-----------------------')
        # # print('logits',logits)
        # # print('not_mask',not_mask)
        # print('target',target)
        # print(target.shape)
        # print(output['pre_logits'])
        # print(output['pre_logits'].shape)
        # print('-----------------------')
        # exit(0)

        if model.embeddings is None:
            model.embeddings = output['pre_logits']
            model.targets = target
            # print('target:',target)
        else:
            model.embeddings = torch.cat((model.embeddings, output['pre_logits']), dim=0)
            # print('target:',target)
            model.targets = torch.cat((model.targets, target), dim=0)


        # loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        
        # if args.pull_constraint and 'reduce_sim' in output:
        #     loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        # acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        # if not math.isfinite(loss.item()):
        #     print("Loss is {}, stopping training".format(loss.item()))
        #     sys.exit(1)

        # optimizer.zero_grad()
        # loss.backward() 
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # optimizer.step()

        # torch.cuda.synchronize()
        # metric_logger.update(Loss=loss.item())
        # metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        # metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        # metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)

    # torch.set_printoptions(threshold=torch.inf)
    # print('----------------------------------------')
    # # print('Embeddings shape:', model.embeddings.shape)
    # print('Targets:', model.targets)
    # print('----------------------------------------')
    # exit(0)
    unique_targets = torch.unique(model.targets)
    for target in unique_targets:
        target_indices = (model.targets == target).nonzero(as_tuple=False).squeeze()
        # print('target_indices:', target_indices.shape)
        target_embeddings = model.embeddings[target_indices]
        # print('target_embeddings:', target_embeddings.shape)
        target_mean = torch.mean(target_embeddings, dim=0)
        # print('target_mean_shape', target_mean.shape)
        # print(torch.norm(target_mean).shape)
        target_mean = target_mean / torch.norm(target_mean)
        model.mean_embeddings[target] = target_mean
    # print('Mean embeddings shape:', model.mean_embeddings)
    # print('----------------------------------------')
    # print(model.head.weight)
    # print(model.head.weight.shape)
    # print(model.head.bias)
    # print('----------------------------------------')
    # exit(0)

    for i in range(args.nb_classes):
        model.ncm_head.weight.data[i] = model.mean_embeddings[i]
        
        # model.head.weight.data[i] = model.mean_embeddings[i]
    model.embeddings = None
    model.targets = None
    # print(model.head.weight)

    return

def replace_with_match_ncm(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):

    model.eval()
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            if args.multi_query==False:
                if original_model is not None:
                    output = original_model(input)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None
            else:
                feature_list= []
                for i in range(task_id+1):
                    output = model(input, task_id=i,query=True)
                        
                        
                    feature_list.append(output['pre_logits'])
                for i in range(task_id+1, args.num_tasks):
                    feature_list.append(torch.zeros((input.shape[0], 768)).to(device))
                        # feature_list.append(torch.zeros((input.shape[0], 768)))
                cls_features = torch.stack(feature_list, dim=1)

            
            output = model(input, task_id=task_id, cls_features=cls_features)


        # print('-----------------------')
        # # print('logits',logits)
        # # print('not_mask',not_mask)
        # print('target',target)
        # print(target.shape)
        # print(output['pre_logits'])
        # print(output['pre_logits'].shape)
        # print('-----------------------')
        # exit(0)

        if model.embeddings is None:
            model.embeddings = output['pre_logits']
            model.targets = target
            # print('target:',target)
        else:
            model.embeddings = torch.cat((model.embeddings, output['pre_logits']), dim=0)
            # print('target:',target)
            model.targets = torch.cat((model.targets, target), dim=0)


        # loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        
        # if args.pull_constraint and 'reduce_sim' in output:
        #     loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        # acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        # if not math.isfinite(loss.item()):
        #     print("Loss is {}, stopping training".format(loss.item()))
        #     sys.exit(1)

        # optimizer.zero_grad()
        # loss.backward() 
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # optimizer.step()

        # torch.cuda.synchronize()
        # metric_logger.update(Loss=loss.item())
        # metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        # metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        # metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)

    # torch.set_printoptions(threshold=torch.inf)
    # print('----------------------------------------')
    # # print('Embeddings shape:', model.embeddings.shape)
    # print('Targets:', model.targets)
    # print('----------------------------------------')
    # exit(0)
    unique_targets = torch.unique(model.targets)
    for target in unique_targets:
        target_indices = (model.targets == target).nonzero(as_tuple=False).squeeze()
        # print('target_indices:', target_indices.shape)
        target_embeddings = model.embeddings[target_indices]
        # print('target_embeddings:', target_embeddings.shape)
        target_mean = torch.mean(target_embeddings, dim=0)
        # print('target_mean_shape', target_mean.shape)
        # print(torch.norm(target_mean).shape)
        target_mean = target_mean / torch.norm(target_mean)
        model.mean_embeddings[target] = target_mean


    for i in range(args.nb_classes):
        model.ncm_head.weight.data[i] = model.mean_embeddings[i]
    
        # model.head.weight.data[i] = model.mean_embeddings[i]
    model.embeddings = None
    model.targets = None
    # print(model.head.weight)

    return

def replace_with_key(model: torch.nn.Module):
    keys=model.e_prompt.prompt_key.reshape(-1,768)
    # print('keys:',torch.norm(keys,dim=1).shape)
    # print('keys:',torch.norm(keys,dim=1,keepdim=True).shape)
    keys=keys/torch.norm(keys,dim=1,keepdim=True)
    # print('keys_normed:',keys.shape)
    # print('model.head.weight:',model.head.weight.shape)
    
    model.ncm_head.weight.data=keys
    
    # model.head.weight.data=keys
    return

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        # print(target)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # print('target:',target)
        feature_list = []
        with torch.no_grad():
            if args.multi_query==False:
                if original_model is not None:
                    output = original_model(input)
                    cls_features = output['pre_logits']
                    
                else:
                    cls_features = None
            else:
                for i in range(task_id):
                    feature_list.append(torch.zeros((input.shape[0], 768)).to(device))
                output = model(input, task_id=task_id,query=True)
                feature_list.append(output['pre_logits'])
                for i in range(task_id+1, args.num_tasks):
                    feature_list.append(torch.zeros((input.shape[0], 768)).to(device))
                cls_features = torch.stack(feature_list, dim=1)
                # print(cls_features.shape)
        # print(target)
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode,target=target)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
        # print('-----------------------')
        # # print('logits',logits)
        # # print('not_mask',not_mask)
        # print('target',target)
        # print(target.shape)
        # print(output['pre_logits'])
        # print(output['pre_logits'].shape)
        # print('-----------------------')
        # exit(0)

        # if epoch == args.epochs-1:
        #     if model.embeddings is None:
        #         model.embeddings = output['pre_logits']
        #         model.targets = target
        #         # print('target:',target)
        #     else:
        #         model.embeddings = torch.cat((model.embeddings, output['pre_logits']), dim=0)
        #         # print('target:',target)
        #         model.targets = torch.cat((model.targets, target), dim=0)


        # # ʾ��ʹ��
        # if is_distributed_training():
        #     print("using distributed training")
        # else:
        #     print("not using distributed training")

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # if epoch == args.epochs-1:
    #     torch.set_printoptions(threshold=torch.inf)
    #     # print('----------------------------------------')
    #     # # print('Embeddings shape:', model.embeddings.shape)
    #     # print('Targets:', model.targets)
    #     # print('----------------------------------------')
    #     # exit(0)
    #     unique_targets = torch.unique(model.targets)
    #     for target in unique_targets:
    #         target_indices = (model.targets == target).nonzero(as_tuple=False).squeeze()
    #         # print('target_indices:', target_indices.shape)
    #         target_embeddings = model.embeddings[target_indices]
    #         # print('target_embeddings:', target_embeddings.shape)
    #         target_mean = torch.mean(target_embeddings, dim=0)
    #         target_mean = target_mean / torch.norm(target_mean)
    #         model.mean_embeddings[target] = target_mean
    #     # print('Mean embeddings shape:', model.mean_embeddings)
    #     # print('----------------------------------------')
    #     # print(model.head.weight)
    #     # print(model.head.weight.shape)
    #     # print(model.head.bias)
    #     # print('----------------------------------------')
    #     # exit(0)

    #     # for i in range(args.nb_classes):
    #     #     model.head.weight.data[i] = model.mean_embeddings[i]
    #     model.embeddings = None
    #     model.targets = None
        # print(model.head.weight)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,subtask_id=-1):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(subtask_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()
    matching_num=0
    matching_all=0

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # print('target:',target)
            # print(target)
            # print("task_id",task_id)
            # compute output
            if args.multi_query==False:
                if original_model is not None:
                    output = original_model(input)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None
            else:
                feature_list= []
                for i in range(task_id+1):
                    output = model(input, task_id=i,query=True)
                        
                    
                    feature_list.append(output['pre_logits'])
                for i in range(task_id+1, args.num_tasks):
                    feature_list.append(torch.zeros((input.shape[0], 768)).to(device))
                    # feature_list.append(torch.zeros((input.shape[0], 768)))
                cls_features = torch.stack(feature_list, dim=1)

            if model.perfect_match:
                output = model(input, task_id=subtask_id, cls_features=cls_features)
            else:
                output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']


            # print('-----------------------')
            # print('idx',output['prompt_idx'])
            # print('-----------------------')
            matching_num += torch.sum(output['prompt_idx']==subtask_id).item()
            matching_all += output['prompt_idx'].shape[0]

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, matching_num, matching_all


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss
    matching_num_a=0
    matching_all_a=0

    for i in range(task_id+1):
        test_stats, matching_num, matching_all = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=task_id, class_mask=class_mask, args=args,subtask_id=i)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss'] 

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
        print('task',i+1,'Prompt matching ratio: {:.4f}'.format(matching_num/matching_all))
        
        matching_num_a+=matching_num
        matching_all_a+=matching_all
    print('-----------------------------------')
    print('the whole task',task_id+1,'Prompt matching ratio: {:.4f}'.format(matching_num_a/matching_all_a))
    
    print('-----------------------------------')
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.e_prompt.prompt.grad.zero_()
                            model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.e_prompt.prompt.grad.zero_()
                            model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.e_prompt.prompt_key.grad.zero_()
                        model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.e_prompt.prompt_key.grad.zero_()
                        model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        for epoch in range(args.epochs):            
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
            
            if lr_scheduler:
                lr_scheduler.step(epoch)
        if args.NCM:
            print('--------------------------------------')
            print('using NCM')
            replace_with_ncm(model=model, original_model=original_model, criterion=criterion, 
                                            data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                            device=device, epoch=epoch, max_norm=args.clip_grad, 
                                            set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
            print('--------------------------------------')
        
        if args.KEY_replace and args.multi_query and args.multi_key:
            print('--------------------------------------')
            print('using KEY_replace')
            replace_with_key(model=model)
            print('--------------------------------------')
        
        
        
        if args.Match_NCM:
            print('--------------------------------------')
            print('using Match_NCM')
            replace_with_match_ncm(model=model, original_model=original_model, criterion=criterion, 
                                            data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                            device=device, epoch=epoch, max_norm=args.clip_grad, 
                                            set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
            print('--------------------------------------')


        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
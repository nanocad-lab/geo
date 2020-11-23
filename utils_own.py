import torch

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import time
from multiprocessing import Process

from utils import *

from sys import exit

scale_factor = 128.0

def set_grad(params, params_with_grad):
    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(param.data.new().resize_(*param.data.size()))
        if param_w_grad.grad is not None:
            param.grad.data.copy_(param_w_grad.grad.data)
            
def set_data(model, model_with_data):
    for param, param_w_data in zip(model.parameters(), model_with_data.parameters()):
        if param_w_data.data is not None:
            param.data.copy_(param_w_data.data)
        
def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None, verbal=False, iteration=None, param_copy=None, quantize=False, modules=None, use_target=False, monitor=False, mult_pass=False, train_s=False, nas=False):
    '''
    Copied from BinarizedNN.pytorch rep
    '''
    global scale_factor
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    torch.cuda.empty_cache()
    end = time.time()
    
    acc = 0
    prop = 1
    
    if modules is None:
        modules = model
    
    if isinstance(model, nn.DataParallel):
        parallel = True
    else:
        parallel = False
    
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if iteration is not None:
            if i >= iteration:
                break

        # measure data loading time.
        if not parallel:
            inputs = inputs.to(dtype=model.fc1.weight.dtype)
            target = target.to(model.fc1.weight.device, non_blocking=True)
            input_var = inputs.to(model.fc1.weight.device, non_blocking=True)
        else:
            inputs = inputs.to(dtype=model.module.fc1.weight.dtype)
            target = target.to(model.module.fc1.weight.device, non_blocking=True)
            input_var = inputs.to(model.module.fc1.weight.device, non_blocking=True)
        # compute output

        if use_target:
            output = model(input_var, target)
        else:
            output = model(input_var)

        loss = criterion(output, target) * scale_factor
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        
        acc += inputs.size(0)
            
        if training:
            # compute gradient and do SGD step
            for param in modules.parameters():
                param.grad = None
            if train_s:
                loss_long.backward(retain_graph=True)
            loss.backward()
            if acc>=128:
                for p in modules.modules():
                    if hasattr(p, 'weight_org'):
                        p.weight.data.copy_(p.weight_org)
                set_grad(param_copy, list(modules.parameters()))
                if scale_factor != 1:
                    for param in param_copy:
                        param.grad.data = param.grad.data/scale_factor
                optimizer.step()
                params = list(modules.parameters())
                for j in range(len(params)):
                    params[j].data.copy_(param_copy[j].data)
                for p in modules.modules():
                    if hasattr(p, 'weight_org'):
                        p.weight_org.copy_(p.weight.data.clamp(-1,0.999))
                optimizer.zero_grad()
                acc = 0
                        
        batch_time.update(time.time() - end)
        end = time.time()
        if verbal:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.6f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i+1, len(data_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))                    
        del inputs, input_var, target, output, loss, prec1, prec5

    if training:
        for p in model.modules():
            if isinstance(p, nn.Linear) or isinstance(p, nn.Conv2d):
                p.weight.data.clamp_(-1,0.999)
    if not training:
        if verbal:
            print('Epoch: [{0}]\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer, param_copy, iteration=None, quantize=False, modules=None, target=False, monitor=False, mult_pass=False, train_s=False, nas=False):
    '''
    Copied from BinarizedNN.pytorch rep. Iteration specifies the number of batches to run (to save time)
    '''
    # switch to train mode
    model.train()
    optimizer.zero_grad()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer, iteration=iteration, param_copy=param_copy, quantize=quantize, modules=modules, use_target=target, monitor=monitor, mult_pass=mult_pass, train_s=train_s, nas=nas)


def validate(data_loader, model, criterion, epoch, verbal=False, quantize=False, modules=None, target=False, monitor=False, mult_pass=False, train_s=False, nas=False):
    '''
    Copied from BinarizedNN.pytorch rep. Verbal allows training information to be displayed. Iteration specifies the number of batches to run (to save time)
    '''
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        return forward(data_loader, model, criterion, epoch, training=False, optimizer=None, verbal=verbal, quantize=quantize, modules=modules, use_target=target, monitor=monitor, mult_pass=mult_pass, train_s=train_s, nas=nas)
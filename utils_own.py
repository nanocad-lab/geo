import os
import torch
import logging
import torchvision
import torchvision.transforms as transforms

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import time
from multiprocessing import Process

from utils import *

from sys import exit

scale_factor = 128.0
compute_type = torch.float32
compare_type = torch.float32
use_or = True # Takes precedence
use_sync = False
global_sync = 7
# monitoring = True
global_noerr = False
global_share = False
global_split = 1
global_prec = 128
global_lfsr = True
global_usesync = False
global_mixadd = False

limit = 1

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}



def bitstream_sync(a, sync=1, prec=8):
    bit_length = 2**prec
    sec_length = 2**sync
    
    a_scale = np.round(a*bit_length)
    
    seed = np.random.permutation(sec_length)
    scale = 2**np.arange(0,prec,sync)
    
    a_size = list(np.shape(a))
    a_size.append(prec//sync)
    a_size = tuple(a_size)
    
    res_pre = []
    for i in range(bit_length//sec_length):
        pos = np.random.randint(0, sec_length, a_size)
        for j in range(sec_length):
            val = np.sum(seed[pos] * scale, -1)
            res_pre.append(a_scale > val)
            pos = (pos+1)%sec_length
    return np.array(res_pre)

def bitstream(a, bit_length):
    a_size = a.size()
    device = a.device
    
    a_scale = (a*bit_length).to(compare_type)
    output = []
    for i in range(bit_length):
        rand = torch.randint(0,bit_length, a_size, device=device).to(compare_type)
        bit = (a_scale > rand).to(compare_type)
        output.append(bit)
    output = torch.stack(output, 0).to(a.dtype)
    return output

def bitstream_acc(a, bit_length):
    a_size = a.size()
    device = a.device
    
    a_scale = (a*bit_length).to(compare_type)
    output = []
    for i in range(bit_length):
        a_actual = a_scale / bit_length
        rand = torch.rand_like(a_actual)
        bit = (a_actual > rand).to(compare_type)
        output.append(bit)
        a_scale -= bit
        bit_length -= 1
    output = torch.stack(output, 0).to(a.dtype)
    return output

def bitstream_def(a, seed):
    bit_length = len(seed)
    
    a_scale = np.round(a*bit_length)
    a_size = np.shape(a)
    
    pos = np.zeros(a_size).astype(int)
    res_pre = []
    for i in range(bit_length):
        val = seed[pos]
        res_pre.append(a_scale > val)
        pos = (pos+1)%bit_length
    return np.array(res_pre)

def lse(A, b, B, d):
    """
    Equality-contrained least squares.
    The following algorithm minimizes ||Ax - b|| subject to the
    constrain Bx = d.
    Parameters
    ----------
    A : array-like, shape=[m, n]
    B : array-like, shape=[p, n]
    b : array-like, shape=[m]
    d : array-like, shape=[p]
    Reference
    ---------
    Matrix Computations, Golub & van Loan, algorithm 12.1.2
    Examples
    --------
    >>> A = np.array([[0, 1], [2, 3], [3, 4.5]])
    >>> b = np.array([1, 1])
    >>> # equality constrain: ||x|| = 1.
    >>> B = np.ones((1, 3))
    >>> d = np.ones(1)
    >>> lse(A.T, b, B, d)
    array([-0.5,  3.5, -2. ])
    """
    from scipy import linalg
    if not hasattr(linalg, 'solve_triangular'):
        # compatibility for old scipy
        solve_triangular = linalg.solve
    else:
        solve_triangular = linalg.solve_triangular
    A, b, B, d = map(np.asanyarray, (A, b, B, d))
    p = B.shape[0]
    Q, R = linalg.qr(B.T)
    y = solve_triangular(R[:p, :p].T, d)
    A = np.dot(A, Q)
    z = linalg.lstsq(A[:, p:], b - np.dot(A[:, :p], y))[0].ravel()
    return np.dot(Q[:, :p], y) + np.dot(Q[:, p:], z)

def calculate_coef(activation_pos_cor, activation_neg_cor, output_or):
    output_or_pos = output_or[0].reshape(-1)
    output_or_neg = output_or[1].reshape(-1)
    output_or = torch.cat([output_or_pos, output_or_neg], 0)
    activation_cor = torch.cat([activation_pos_cor.reshape(-1), activation_neg_cor.reshape(-1)], 0)
    activation_exp = 1 - torch.exp(-activation_cor)
    activation_tanh = torch.tanh(activation_cor)
    activation_stack = torch.stack([activation_exp, activation_tanh], 1)
    B = np.array([[1,1]])
    d = np.array([1])
    coefs = lse(activation_stack.cpu().numpy(), output_or.cpu().numpy(), B, d)
    return coefs[0]

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
    weight=[0.7,0.3]
    if mult_pass:
        top1_correct_positive = AverageMeter()
        top1_false_positive = AverageMeter()
        top1_correct_negative = AverageMeter()
        top1_false_negative = AverageMeter()
        criterion_skew = nn.CrossEntropyLoss(weight=torch.Tensor(weight)).to(model.fc1.weight.device)
    torch.cuda.empty_cache()
    end = time.time()
    
    acc = 0
    prop = 1
    
    if modules is None:
        modules = model
    
    if isinstance(model, nn.DataParallel):
        parallel = True
    #    add_or = model.module.add_or
    else:
        parallel = False
    #    add_or = model.add_or
    
    for i, (inputs, target) in enumerate(data_loader):
        
#         try:
#             dump_v = np.load("target_svhn.npy")
#         except:
#             dump_v = target.detach().cpu().numpy()
#         else:
#             dump_c = target.detach().cpu().numpy()
#             dump_v = np.concatenate((dump_v, dump_c), axis=0)
#         np.save("target_svhn.npy", dump_v)
        # measure data loading time
        data_time.update(time.time() - end)

        if iteration is not None:
            if i >= iteration:
                break

        # measure data loading time.
        if inputs.size(-1)==224:
            target -= 1
#         try:
        if not parallel:
            inputs = inputs.to(dtype=model.fc1.weight.dtype)
            target = target.to(model.fc1.weight.device, non_blocking=True)
#             np.save("./dump_dump/target.npy", target.data.numpy())
            input_var = inputs.to(model.fc1.weight.device, non_blocking=True)
     #       if training and (i>=prop*len(data_loader)):
     #           model.add_or = True
#         except:
        else:
            inputs = inputs.to(dtype=model.module.fc1.weight.dtype)
            target = target.to(model.module.fc1.weight.device, non_blocking=True)
#             np.save("./dump_dump/target.npy", target.data.numpy())
            input_var = inputs.to(model.module.fc1.weight.device, non_blocking=True)
     #       if training and (i>=prop*len(data_loader)):
     #           model.module.add_or = True
        # compute output

        if use_target:
            output = model(input_var, target)
        else:
            output = model(input_var)
          
        if nas:
            output, output_cost = output
            criterion_cost = nn.MSELoss()
            loss = criterion(output, target)*scale_factor
            loss_cost = criterion_cost(output_cost, torch.tensor(0.).to(output_cost.device))*scale_factor/10000
            loss = loss + loss_cost  
        elif mult_pass:
            output_0 = output[:,:model.num_classes].float()
            output_1 = output[:,model.num_classes:model.num_classes+2].float()
            output_2 = output[:,-model.num_classes:].float()
#             loss = criterion(output_0, target)
#             loss = loss * scale_factor
            _, pred = output_0.topk(1,1,True,True)
            _, pred_pred = output_1.topk(1,1,True,True)
            _, pred_2 = output_2.topk(1,1,True,True)
            pred = pred.t()[0].data
            pred_pred = pred_pred.t()[0].data
            
            pred_2 = pred_2.t()[0].data
            target_pred = (pred==pred_2).long()
#             target_pred = (pred>4).long()
            loss_pred = criterion_skew(output_1, target_pred)
            loss_pred = loss_pred*scale_factor
#             pred_pred = torch.rand_like(pred_pred.float())>0.19
#             loss_pred = 0
#             output = output_2.clone()
#             output[pred_pred.view(-1,1)] = output_0[pred_pred.view(-1,1)]
            if train_s:
#                 print("Here")
                output = output_0
                loss_long = criterion(output_2, target) * scale_factor * 0.2
#                 output = output_0 * 0.9 + output_2 * 0.1
#                 output.data = output_0.data
#                 weight_0 = F.softmax(output_1, dim=-1).data.clone().detach()
#                 output = output_0 * weight_0[:,0:1] + output_2 * weight_0[:,1:2]
#                 output.data = (output_0*(pred_pred.view(-1,1)).float() + output_2*((1-pred_pred.float()).view(-1,1))).data
            else:
                output_0 = output_0*(pred_pred.view(-1,1)).float()
                output_2 = output_2*((1-pred_pred.float()).view(-1,1))
                output = output_0 + output_2
            loss = criterion(output, target) * scale_factor + loss_pred
#             loss_0 = criterion(output_0, target) * scale_factor
#             loss_2 = criterion(output_2, target) * scale_factor
#             loss = loss_0 + loss_pred + loss_2
            cur_correct_pos = (target_pred*(pred_pred==1)).float().mean()*100
            cur_false_pos = ((1-target_pred)*(pred_pred==1)).float().mean()*100
            cur_correct_neg = (target_pred*(pred_pred==0)).float().mean()*100
            cur_false_neg = ((1-target_pred)*(pred_pred==0)).float().mean()*100
            top1_correct_positive.update(cur_correct_pos, inputs.size(0))
            top1_false_positive.update(cur_false_pos, inputs.size(0))
            top1_correct_negative.update(cur_correct_neg, inputs.size(0))
            top1_false_negative.update(cur_false_neg, inputs.size(0))
        else:
            loss = criterion(output, target) * scale_factor
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

#         losses.update(loss.data[0], inputs.size(0))
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
#                 loss_pred.backward(retain_graph=True)
            loss.backward()
    #        print(model.fc1.weight.grad)
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
#             print(model.bn1_1.weight)
                        
        batch_time.update(time.time() - end)
        end = time.time()
        if training or verbal:
            if (i%100==99) and (inputs.size(2)>=64):
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i+1, len(data_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
            elif monitor:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.6f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i+1, len(data_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))                    
        del inputs, input_var, target, output, loss, prec1, prec5
        #exit(0)

    if training:
        for p in model.modules():
            if isinstance(p, nn.Linear) or isinstance(p, nn.Conv2d):
                p.weight.data.clamp_(-1,0.999)
    if not training:
        if verbal:
            if mult_pass:
                print('Epoch: [{0}]\t'
                             'Loss {loss.avg:.4f}\t'
                             'Prec@1 {top1.avg:.3f}\t'
                             'Cor pos {top1_correct_positive.avg:.3f}\t'
                             'Fal pos {top1_false_positive.avg:.3f}\t'
                             'Cor neg {top1_correct_negative.avg:.3f}\t'
                             'Fal neg {top1_false_negative.avg:.3f}\t'.format(
                                 epoch, loss=losses, top1=top1, top1_correct_positive=top1_correct_positive,
                                 top1_false_positive=top1_false_positive, top1_correct_negative=top1_correct_negative,
                                 top1_false_negative=top1_false_negative))
            else:
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

def rms(input):
    return torch.sqrt(torch.mean(input**2))

class OrAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = 1-torch.prod(1-input, dim=-1)
        ctx.size = input.size()
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_output = grad_output.unsqueeze(-1)
        grad_input = grad_output * torch.ones(ctx.size, device=grad_output.device)
        return grad_input
    
orAdd = OrAdd.apply


def linear_quant(activation, weight, err):
    err_mult = 2**err
    weight_pos = weight.clamp(0,100)
    weight_neg = -weight.clamp(-100,0)
    if len(activation.size())==3:
        weight_pos = weight_pos.unsqueeze(-1)
        weight_neg = weight_neg.unsqueeze(-1)
    
    activation_s = activation.unsqueeze(1)
    
    activation_pos_mult = activation_s * weight_pos
    activation_neg_mult = activation_s * weight_neg
    
    activation_pos_mult.data = quantize(activation_pos_mult.data, quant=True, prec=err)
    activation_neg_mult.data = quantize(activation_neg_mult.data, quant=True, prec=err)
    activation_pos = activation_pos_mult.sum(dim=2)
    activation_neg = activation_neg_mult.sum(dim=2)
    return activation_pos - activation_neg

def linear_or_reduce(activation, weight_pos, weight_neg):
    '''
    image (B, Cin) or (B, Bsub, Cin)
    weight (Cout, Cin)
    '''
    if len(activation.size())==3:
        weight_pos = weight_pos.unsqueeze(-1)
        weight_neg = weight_neg.unsqueeze(-1)
    activation_s = activation.unsqueeze(1)
    activation_pos = 1-torch.prod(1-activation_s * weight_pos, dim=2)
    activation_neg = 1-torch.prod(1-activation_s * weight_neg, dim=2)
    return activation_pos, activation_neg

def linear_or_stream_both(activation, weight):
    a, a_v = activation
    w, w_v = weight
    device = a.device
    bit_length = a.size(0)
    bit_range = bit_length-1
    
    result = []
    
    for i in range(bit_length):
        result.append(F.linear(a[i], w[i]).sign())
    result = torch.stack(result, 0)
    result_v = 1-torch.exp(-F.linear(a_v, w_v))
    result_v.data = result.mean(0)
    return (result, result_v)

def linear_or_stream(activation, weight):
    prec = global_prec
    a, a_v = activation
    w = weight
    device = a.device
    bit_length = a.size(0)
    bit_range = prec-1
    
    w_pos = (weight*bit_length).clamp(0,bit_range).to(compare_type)
    w_neg = -(weight*bit_length).clamp(-bit_range,0).to(compare_type)
    
    if global_share:
        w_size = [w.size(-1)]
    else:
        w_size = w.size()
        
    if global_lfsr:
        weight_split_size_flat = np.array(w_size).prod()
        weight_seed_pos = np.arange(67, weight_split_size_flat+67)%(prec-1)+1
        weight_seed_neg = np.arange(37, weight_split_size_flat+37)%(prec-1)+1
        rand_weight_pos = torch.from_numpy(weight_seed_pos).reshape(w_size).to(device)
        rand_weight_neg = torch.from_numpy(weight_seed_neg).reshape(w_size).to(device)
    elif global_usesync:
        seed = torch.from_numpy(np.random.permutation(prec)).to(device)
        weight_pos_pos = torch.randint(prec, w_size, dtype=torch.int64, device=device)
        weight_neg_pos = torch.randint(prec, w_size, dtype=torch.int64, device=device)
    else:
        rand_weight_pos = torch.randint(prec, w_size, dtype=compare_type, device=device)
        rand_weight_neg = torch.randint(prec, w_size, dtype=compare_type, device=device)
        
    result_pos = []
    result_neg = []
    
    for j in range(0, bit_length):
        if global_lfsr:
            rand_weight_pos = ((rand_weight_pos//32)%2+rand_weight_pos//64)%2+2*(rand_weight_pos%64)
            rand_weight_neg = ((rand_weight_neg//32)%2+rand_weight_neg//64)%2+2*(rand_weight_neg%64)
        elif global_usesync:
            rand_weight_pos = seed[weight_pos_pos]
            rand_weight_neg = seed[weight_neg_pos]
            weight_pos_pos = (weight_pos_pos+1)%prec
            weight_neg_pos = (weight_neg_pos+1)%prec
        else:
            rand_weight_pos.random_(0,prec)
            rand_weight_neg.random_(0,prec)
        w_pos_bit = (w_pos > rand_weight_pos).to(compute_type)
        w_neg_bit = (w_neg > rand_weight_neg).to(compute_type)
        result_pos.append(F.linear(a[j], w_pos_bit).sign())
        result_neg.append(F.linear(a[j], w_neg_bit).sign())
        
    result_pos = torch.stack(result_pos, 0)
    result_neg = torch.stack(result_neg, 0)
    
    w_pos = weight.clamp(0,100)
    w_neg = -(weight.clamp(-100,0))
    a_v = a_v.to(w_pos.dtype)
    result_pos_value = 1-torch.exp(-F.linear(a_v, w_pos))
    result_pos_value.data = result_pos.mean(0)
    result_neg_value = 1-torch.exp(-F.linear(a_v, w_neg))
    result_neg_value.data = result_neg.mean(0)
    
    return (result_pos, result_pos_value), (result_neg, result_neg_value)

def linear_or_shared(activation, weight, bit_length, split=1):
    device = activation.device
    bit_range = bit_length - 1
    
    split_size = activation.size(1)//split
    device = activation.device
    
    input_split = torch.split((activation*bit_length).to(compare_type), split_size, 1)
    w_pos_split = torch.split((weight*bit_length).clamp(0,bit_range).to(compare_type), split_size, 1)
    w_neg_split = torch.split(-(weight*bit_length).clamp(-bit_range,0).to(compare_type), split_size, 1)
    
    if global_share:
        input_split_size = input_split[0].size(-1)
        weight_split_size= w_pos_split[0].size(-1)
        rand_input = torch.randint(bit_range, [input_split_size], dtype=compare_type, device=device)
        rand_weight_pos = torch.randint(bit_range, [weight_split_size], dtype=compare_type, device=device)
        rand_weight_neg = torch.randint(bit_range, [weight_split_size], dtype=compare_type, device=device)
    else:
        input_split_size = input_split[0].size()
        weight_split_size= w_pos_split[0].size()
        rand_input = torch.randint(bit_range, input_split_size, dtype=compare_type, device=device)
        rand_weight_pos = torch.randint(bit_range, weight_split_size, dtype=compare_type, device=device)
        rand_weight_neg = torch.randint(bit_range, weight_split_size, dtype=compare_type, device=device)
    
    input_bit = (input_split[0] > rand_input).to(compute_type)
    w_pos_bit = (w_pos_split[0] > rand_weight_pos).to(compute_type)
    w_neg_bit = (w_neg_split[0] > rand_weight_neg).to(compute_type)
    result_pos = F.linear(input_bit, w_pos_bit).sign()
    result_neg = F.linear(input_bit, w_neg_bit).sign()
    
    for i in range(1, split):
        rand_input.random_(0, bit_range)
        rand_weight_pos.random_(0, bit_range)
        rand_weight_neg.random_(0, bit_range)
        input_bit = (input_split[i] > rand_input).to(compute_type)
        w_pos_bit = (w_pos_split[i] > rand_weight_pos).to(compute_type)
        w_neg_bit = (w_neg_split[i] > rand_weight_neg).to(compute_type)
        result_pos += F.linear(input_bit, w_pos_bit).sign()
        result_neg += F.linear(input_bit, w_neg_bit).sign()
    
    for j in range(1, bit_length):
        for i in range(split):
            rand_input.random_(0, bit_range)
            rand_weight_pos.random_(0, bit_range)
            rand_weight_neg.random_(0, bit_range)
            input_bit = (input_split[i] > rand_input).to(compute_type)
            w_pos_bit = (w_pos_split[i] > rand_weight_pos).to(compute_type)
            w_neg_bit = (w_neg_split[i] > rand_weight_neg).to(compute_type)
            result_pos += F.linear(input_bit, w_pos_bit).sign()
            result_neg += F.linear(input_bit, w_neg_bit).sign()
            
    result_pos_scale = result_pos / bit_length
    result_neg_scale = result_neg / bit_length
    return result_pos_scale, result_neg_scale

def linear_or_split(activation, weight, output_pos, output_neg, err=7, split=4):
    bit_length = 2**err
    bit_range = bit_length - 1
    
    split_size = activation.size(1)/split
    device = activation.device
    
    input_split = torch.split((activation*bit_length).to(compute_type), split_size, 1)
    w_pos_split = torch.split((weight*bit_length).clamp(0,bit_range).to(compute_type), split_size, 1)
    w_neg_split = torch.split(-(weight*bit_length).clamp(-bit_range,0).to(compute_type), split_size, 1)
    
    input_split_size = input_split[0].size()
    weight_split_size= w_pos_split[0].size()
    rand_input = torch.randint(bit_range, input_split_size, device=device)
    rand_weight_pos = torch.randint(bit_range, weight_split_size, device=device)
    rand_weight_neg = torch.randint(bit_range, weight_split_size, device=device)
    
    input_bit = (input_split[0] > rand_input).to(compute_type)
    w_pos_bit = (w_pos_split[0] > rand_weight_pos).to(compute_type)
    w_neg_bit = (w_neg_split[0] > rand_weight_neg).to(compute_type)
    result_pos = F.linear(input_bit, w_pos_bit).sign()
    result_neg = F.linear(input_bit, w_neg_bit).sign()
    
    for i in range(1, split):
        rand_input.random_(0, bit_range).round_()
        rand_weight_pos.random(0, bit_range).round_()
        rand_weight_neg.random(0, bit_range).round_()
        input_bit = (input_split[i] > rand_input).to(compute_type)
        weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(compute_type)
        weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(compute_type)
        result_pos += F.linear(input_bit, w_pos_bit).sign()
        result_neg += F.linear(input_bit, w_neg_bit).sign()
    
    for j in range(1, bit_length):
        for i in range(split):
            rand_input.random_(0, bit_range).round_()
            rand_weight_pos.random(0, bit_range).round_()
            rand_weight_neg.random(0, bit_range).round_()
            input_bit = (input_split[i] > rand_input).to(compute_type)
            weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(compute_type)
            weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(compute_type)
            result_pos += F.linear(input_bit, w_pos_bit).sign()
            result_neg += F.linear(input_bit, w_neg_bit).sign()
            
    result_pos_scale = result_pos / bit_length
    result_neg_scale = result_neg / bit_length
    return result_pos_scale, result_neg_scale

def linear_or_acc(activation, weight, output_pos, output_neg, err=7, sync=global_sync):
    bit_length = 2**err
    sec_length = 2**sync
    
    bit_range = bit_length - 1
    
    a_scale = (activation*bit_length).to(compute_type)
    w_pos_scale = (weight*bit_length).clamp(0,bit_range).to(compute_type)
    w_neg_scale = -(weight*bit_length).clamp(-bit_range,0).to(compute_type)
    
    seed = torch.from_numpy(np.tile(np.random.permutation(sec_length),2)).to(activation.device, compute_type)
    scale = torch.from_numpy(2**np.arange(0,err,sync)).to(activation.device, compute_type)
    
    a_size = list(activation.size())
    w_size = list(weight.size())
    
    a_size.append(err//sync)
    w_size.append(err//sync)
    
    a_size = tuple(a_size)
    w_size = tuple(w_size)
    
    for i in range(bit_length//sec_length):
        pos_a = torch.randint(0, sec_length, a_size, device=activation.device)
        pos_w_pos = torch.randint(0, sec_length, w_size, device=activation.device)
        pos_w_neg = torch.randint(0, sec_length, w_size, device=activation.device)
        for j in range(sec_length):
            val_a = torch.sum(seed[pos_a] * scale, -1).to(compute_type)
            val_w_pos = torch.sum(seed[pos_w_pos] * scale, -1).to(compute_type)
            val_w_neg = torch.sum(seed[pos_w_neg] * scale, -1).to(compute_type)
            a_bit = (a_scale > val_a).to(compute_type)
#             print(a_bit)
            w_pos_bit = (w_pos_scale > val_w_pos).to(compute_type)
            w_neg_bit = (w_neg_scale > val_w_neg).to(compute_type)
            
            output_pos += F.linear(a_bit, w_pos_bit).sign()
            output_neg += F.linear(a_bit, w_neg_bit).sign()
            
            pos_a = (pos_a+1)
            pos_w_pos = (pos_w_pos+1)
            pos_w_neg = (pos_w_neg+1)
            
    output_pos /= bit_length
    output_neg /= bit_length
    return torch.stack((output_pos, output_neg))         

def linear_or_subtract(activation, weight, add_or=False, convert=False, err=16, add_full=1, true_or=False, add_count=False, noerr=False or global_noerr):
    '''
    image (B, Cin) or (B, Bsub, Cin)
    weight (Cout, Cin)
    '''
    err_mult = 2**err
    weight_pos = weight.clamp(0,100)
    weight_neg = -weight.clamp(-100,0)
    
    if add_count:
        activation_pos = F.linear(activation, weight_pos)
        activation_neg = F.linear(activation, weight_neg)
    elif true_or:
        activation_pos, activation_neg = linear_or_reduce(activation, weight_pos, weight_neg)
    else:
        activation_pos_cor = F.linear(activation, weight_pos)
        activation_neg_cor = F.linear(activation, weight_neg)
#         activation_pos_exp = (1-torch.exp(-activation_pos_cor))
#         activation_neg_exp = (1-torch.exp(-activation_neg_cor))
#         activation_pos_tanh = torch.tanh(activation_pos_cor)
#         activation_neg_tanh = torch.tanh(activation_neg_cor)
#         activation_pos = coef*activation_pos_exp + (1-coef)*activation_pos_tanh
#         activation_neg = coef*activation_neg_exp + (1-coef)*activation_neg_tanh
        activation_pos = (1-torch.exp(-activation_pos_cor))
        activation_neg = (1-torch.exp(-activation_neg_cor))
        
    if noerr:
        return activation_pos - activation_neg
        
    if add_count:
        print("Here")
        output = activation_pos - activation_neg
        output_data = sc_extension.linear_count(activation.data, weight.data, err_mult)
        output.data = output_data
    elif add_or:
        output = activation_pos - activation_neg
        if use_or:
#             output_data = sc_extension.linear_or(activation.data, weight.data, err_mult, add_full).to(output.dtype)
            output_data = linear_or_shared(activation.data, weight.data, err_mult)
        elif not use_sync:
            output_data = sc_extension.linear_acc(activation.data, weight.data, err_mult, add_full)
#         coef = calculate_coef(activation_pos_cor, activation_neg_cor, output_data)
        else:
            output_pos = torch.zeros_like(output.data)
            output_neg = torch.zeros_like(output.data)
            output_data = linear_or_acc(activation.data, weight.data, output_pos, output_neg, err)
        output.data = (output_data[0] - output_data[1]).to(output.dtype)
    else:
#         err_mult = err_mult/2
        activation_pos_scale = activation_pos.data
        activation_neg_scale = activation_neg.data
        error_pos = torch.randn_like(activation_pos_scale) * torch.sqrt(activation_pos_scale*(1-activation_pos_scale)/err_mult)
        error_neg = torch.randn_like(activation_neg_scale) * torch.sqrt(activation_neg_scale*(1-activation_neg_scale)/err_mult)
        
        activation_pos.data = activation_pos.data + error_pos
        activation_neg.data = activation_neg.data + error_neg
        output = activation_pos - activation_neg
    return output

def conv2d_quant(activation, weight, padding=0, stride=1, err=16):
    activation_unf = F.unfold(activation, weight.size(2), padding=padding, stride=stride)
    weight_unf = weight.reshape(weight.size(0),-1)
    res_unf = linear_quant(activation_unf,weight_unf,err)
    res = F.fold(res_unf, (activation.size(2), activation.size(3)), (1,1))
    return res

# def avgpool2d_stream(activation, size=2):
#     a, a_v = activation
#     device = a.device
#     dtype = a.dtype
#     bit_length = a.size(0)

#     output_0 = F.avg_pool2d(a[0], size)
#     output_size = output_0.size()
#     rand = torch.rand(output_size, dtype=dtype, device=device)
#     output = []
#     output.append((output_0 > rand).to(dtype))
#     for j in range(1, bit_length):
#         rand = torch.rand(output_size, dtype=dtype, device=device)
#         output.append((F.avg_pool2d(a[j], size)>rand).to(dtype))
#     output = torch.stack(output, 0)
#     output_v = F.avg_pool2d(a_v, size)
#     output_v.data = output.mean(0)
#     return output, output_v
    
def pool2d_sift(activation, sift, size=2):
    output_temp = []
    groups = activation.size(1)
    for i in range(size):
        for j in range(size):
            output_temp.append(F.conv2d(activation[...,i:,j:], sift, stride=size, groups=groups))
    return torch.stack(output_temp, 0)

def avgpool2d_convert(activation, size=2, config="Medium_skip"):
    device = activation[0].device
    dtype = activation[0].dtype
    groups = activation[0].size(1)
    
    sift = torch.ones((groups,1,1,1), dtype=dtype, device=device)
    output = []
    
    if config=="Normal":
        for i in range(size):
            for j in range(size):
                index = i*size+j
                output.append(activation[index])
        output = torch.stack(output, 0).mean(0)
        output = F.avg_pool2d(output, 2)
        
    if config=="Hard_skip":
        for i in range(size):
            for j in range(size):
                index = i*size+j
                output.append(F.conv2d(activation[0][...,i:,j:], sift, stride=size, groups=groups))
        output = torch.stack(output, 0).mean(0)
        
    if config=="Medium_skip":
        for i in range(size):
            for j in range(size):
                index = i
                output.append(F.conv2d(activation[index][...,i:,j:], sift, stride=size, groups=groups))
        output = torch.stack(output, 0).mean(0)
        
    if config=="Soft_skip":
        for i in range(size):
            for j in range(size):
                index = i*size+j
                output.append(F.conv2d(activation[index][...,i:,j:], sift, stride=size, groups=groups))
        output = torch.stack(output, 0).mean(0)
        
    return output
        
def avgpool2d_stream(activation, size=2, config="Hard_skip", shuffle=False):
    a, a_v = activation
    device = a.device
    dtype = a.dtype
    groups = a_v.size(1)
    bit_length = a.size(0)
    
    sift = torch.ones((groups,1,1,1), dtype=dtype, device=device)
    output = []
    if config=="Normal":
        output_0 = pool2d_sift(a[0], sift, size=size).mean(0)
        output_size = output_0.size()
        rand = torch.rand(output_size, dtype=dtype, device=device)
        output.append((output_0 > rand).to(dtype))
        for j in range(1, bit_length):
            rand = torch.rand(output_size, dtype=dtype, device=device)
            output.append((pool2d_sift(a[j], sift, size=size).mean(0)>rand).to(dtype))
        output = torch.stack(output, 0)
    if config=="Hard_skip":
        skip_length = bit_length // (size**2)
        for i in range(size):
            for j in range(size):
                for k in range(skip_length):
                    output.append(F.conv2d(a[k,...,i:,j:], sift, stride=size, groups=groups))
        output = torch.stack(output, 0)
    if config=="Soft_skip":
        skip_length = bit_length // (size**2)
        for i in range(size):
            for j in range(size):
                offset = skip_length*(i*size+j)
                for k in range(skip_length):
                    output.append(F.conv2d(a[offset+k,...,i:,j:], sift, stride=size, groups=groups))
        output = torch.stack(output, 0)
    if config=="Concat":
        skip_length = bit_length
        bit_length *= size**2
        for i in range(size):
            for j in range(size):
                for k in range(skip_length):
                    output.append(F.conv2d(a[k,...,i:,j:], sift, stride=size, groups=groups))
        output = torch.stack(output, 0)
    
    if shuffle:
        output = output.mean(0)
        output = bitstream_acc(output, bit_length)
    output_v = F.avg_pool2d(a_v, size)
    output_v.data = output.mean(0)
    return output, output_v

def unpool(tensor):
    '''
    Input in format (bit_short, pool_size, batch, cin, width/pool_w, height/pool_h)
    Output in format (bit_short, batch, cin, width, height)
    '''
    w = tensor.size(-2)
    h = tensor.size(-1)
    cin = tensor.size(3)
    pool_cnt = tensor.size(1)
    pool_1d = int(np.sqrt(pool_cnt))
    bit_short = tensor.size(0)
    batch = tensor.size(2)
    #First combine the first two rows
    input_sliced = []
    for i in range(pool_1d):
        input_sliced.append([])
        for j in range(pool_1d):
            input_sliced[i].append(tensor[:,i*pool_1d+j].reshape(bit_short,batch,cin,-1))
        input_sliced[i] = torch.stack(input_sliced[i],-1).reshape(bit_short,batch,cin,w,h*pool_1d).permute(0,1,2,4,3).reshape(bit_short,batch,cin,-1)
    input_sliced = torch.stack(input_sliced,-1).reshape(bit_short, batch, cin, h*pool_1d, w*pool_1d).permute(0,1,2,4,3)
    return input_sliced

def subtract_pool_relu(res_pos, res_neg, pool, shuffle=True):
    result_pos, result_pos_value = res_pos
    result_neg, result_neg_value = res_neg
    
    result_pos_unpool = unpool(result_pos)
    result_neg_unpool = unpool(result_neg)
    
def subtract_relu_pool(res_pos, res_neg, pool, shuffle=True):
    result_pos, result_pos_value = res_pos
    result_neg, result_neg_value = res_neg
    
    # Additional synchronization before relu+pool
    result_pos_unpool = unpool(result_pos)
    result_neg_unpool = unpool(result_neg)
    result_pos_value.data = result_pos_unpool.mean(0)
    result_neg_value.data = result_neg_unpool.mean(0)
    
    result = subtract_relu_stream(result_pos, result_neg, warmup=relu_warmup)
#     if shuffle:
#         bit_short = result.size(0)
#         pool_size = result.size(1)
#         sel_size  = result[0,0].size()
#         res_n = []
#         device = result.device
#         seed = torch.from_numpy(np.random.permutation(pool_size)).to(device)
#         rand_pos = torch.randint(pool_size, sel_size, device=device)
#         for i in range(bit_short):
#             res_n_n = []
#             for j in range(pool_size):
# #                 rand_sel = torch.randint(pool_size, sel_size, device=device)
#                 rand_sel = seed[rand_pos]
#                 rand_pos = (rand_pos+1)%pool_size
#                 rand_0 = rand_sel==0
#                 rand_1 = rand_sel==1
#                 rand_2 = rand_sel==2
#                 rand_3 = rand_sel==3
#                 res_c = result[i,j].clone()
#                 res_c[rand_0] = result[i,0][rand_0]
#                 res_c[rand_1] = result[i,1][rand_1]
#                 res_c[rand_2] = result[i,2][rand_2]
#                 res_c[rand_3] = result[i,3][rand_3]
#                 res_n_n.append(res_c)
#             res_n_n = torch.stack(res_n_n, 0)
#             res_n.append(res_n_n)
#         result = torch.stack(res_n, 0)
    if shuffle:
        bit_short = result.size(0)
        device = result.device
        result_c = result[0].clone()
        result_c[:] = result.mean((0,1))
        result = bitstream_acc(result_c, bit_short)
            
    result_value = F.relu(result_pos_value - result_neg_value)
    result_value = F.hardtanh(result_value)
#     result_value = F.softplus(result_pos_value-result_neg_value, beta=12)
    result_unpool = subtract_relu_stream(result_pos_unpool, result_neg_unpool, warmup=relu_warmup)
    result_value.data = result_unpool.mean(0)
    
    result_value = F.avg_pool2d(result_value, pool)
    result_value.data = result.mean((0,1)).float()
    return result, result_value

def conv2d_or_stream_out_pool_prep_mixed(activation, weight, bit_length, padding, stride, pool):
    prec = global_prec
    bit_range = prec - 1
    device = activation.device
    
    # Prescale inputs for bit stream generation
    input_split = (activation.data*prec).to(compare_type)
    input_split = F.pad(input_split, (padding[0], padding[0], padding[1], padding[1]))
    w_pos_split = (weight.data*prec).clamp(0,bit_range).to(compare_type)
    w_neg_split = -((weight.data*prec).clamp(-bit_range,0)).to(compare_type)
    
    if global_share:
        input_split_size = input_split.size()[-3:]
        weight_split_size= w_pos_split.size()[-3:]
    else:
        input_split_size = input_split.size()
        weight_split_size= w_pos_split.size()

    # Currently generation is shared? This can change (relatively easily) by moving it into the computation loop)
    if global_lfsr:
        input_split_size_flat = np.array(input_split_size).prod()
        weight_split_size_flat = np.array(weight_split_size).prod()
        input_seed = np.arange(0, input_split_size_flat)%(prec-1)+1
        weight_seed_pos = np.arange(67, weight_split_size_flat+67)%(prec-1)+1
        weight_seed_neg = np.arange(37, weight_split_size_flat+37)%(prec-1)+1
        rand_input = torch.from_numpy(input_seed).reshape(input_split_size).to(device)
        rand_weight_pos = torch.from_numpy(weight_seed_pos).reshape(weight_split_size).to(device)
        rand_weight_neg = torch.from_numpy(weight_seed_neg).reshape(weight_split_size).to(device)
    elif global_usesync:
        seed = torch.from_numpy(np.random.permutation(prec)).to(device)
        input_pos = torch.randint(prec, input_split_size, dtype=torch.int64, device=device)
        weight_pos_pos = torch.randint(prec, weight_split_size, dtype=torch.int64, device=device)
        weight_neg_pos = torch.randint(prec, weight_split_size, dtype=torch.int64, device=device)
    else:
        rand_input = torch.randint(prec, input_split_size, dtype=compare_type, device=device)
        rand_weight_pos = torch.randint(prec, weight_split_size, dtype=compare_type, device=device)
        rand_weight_neg = torch.randint(prec, weight_split_size, dtype=compare_type, device=device)
    
    result_pos = []
    result_neg = []
    
    pool_cnt = pool[0]*pool[1]
    stride_pool = list(stride[:])
    stride_pool[0] *= pool[0]
    stride_pool[1] *= pool[1]
    
    i_x = input_split.size(2)
    i_y = input_split.size(3)
    f_x = weight.size(2)
    f_y = weight.size(3)
            
    for k in np.arange(0, bit_length, pool_cnt):
        result_pos.append([])
        result_neg.append([])
        if global_lfsr:
            rand_input = ((rand_input//32)%2+rand_input//64)%2+2*(rand_input%64)
            rand_weight_pos = ((rand_weight_pos//32)%2+rand_weight_pos//64)%2+2*(rand_weight_pos%64)
            rand_weight_neg = ((rand_weight_neg//32)%2+rand_weight_neg//64)%2+2*(rand_weight_neg%64)
        elif global_usesync:
            rand_input = seed[input_pos]
            rand_weight_pos = seed[weight_pos_pos]
            rand_weight_neg = seed[weight_neg_pos]
            input_pos = (input_pos+1)%prec
            weight_pos_pos = (weight_pos_pos+1)%prec
            weight_neg_pos = (weight_neg_pos+1)%prec
        else:
            rand_input.random_(0,prec)
            rand_weight_pos.random_(0,prec)
            rand_weight_neg.random_(0,prec)
        input_bit = (input_split > rand_input).to(compute_type)
        w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
        w_neg_bit = (w_neg_split > rand_weight_pos).to(compute_type)
        for i in range(pool[0]):
            for j in range(pool[1]):
                input_bit_cur = input_bit[...,i:,j:]
                result_pos_cur = []
                result_neg_cur = []
                for l in range(f_x):
                    for m in range(f_y):
                        input_bit_cur_cur = input_bit_cur[...,l:i_x-f_x+l+1,m:i_y-f_y+m+1]
                        result_pos_cur.append(F.conv2d(input_bit_cur_cur, w_pos_bit[...,l:l+1,m:m+1], stride=stride_pool).sign())
                        result_neg_cur.append(F.conv2d(input_bit_cur_cur, w_neg_bit[...,l:l+1,m:m+1], stride=stride_pool).sign())
                result_pos_cur = torch.stack(result_pos_cur, 0)
                result_neg_cur = torch.stack(result_neg_cur, 0)
                result_pos[-1].append(result_pos_cur.sum(0))
                result_neg[-1].append(result_neg_cur.sum(0))
        result_pos[-1] = torch.stack(result_pos[-1], 0)
        result_neg[-1] = torch.stack(result_neg[-1], 0)
    result_pos = torch.stack(result_pos, 0)
    result_neg = torch.stack(result_neg, 0)
    
    return result_pos, result_neg

def conv2d_or_stream_out_pool_prep(activation, weight, bit_length, padding, stride, pool):
    prec = global_prec
    bit_range = prec - 1
    device = activation.device
    
    # Prescale inputs for bit stream generation
    input_split = (activation.data*prec).to(compare_type)
    input_split = F.pad(input_split, (padding[0], padding[0], padding[1], padding[1]))
    w_pos_split = (weight.data*prec).clamp(0,bit_range).to(compare_type)
    w_neg_split = -((weight.data*prec).clamp(-bit_range,0)).to(compare_type)
    
    if global_share:
        input_split_size = input_split.size()[-3:]
        weight_split_size= w_pos_split.size()[-3:]
    else:
        input_split_size = input_split.size()
        weight_split_size= w_pos_split.size()

    # Currently generation is shared? This can change (relatively easily) by moving it into the computation loop)
    if global_lfsr:
        input_split_size_flat = np.array(input_split_size).prod()
        weight_split_size_flat = np.array(weight_split_size).prod()
        input_seed = np.arange(0, input_split_size_flat)%(prec-1)+1
        weight_seed_pos = np.arange(67, weight_split_size_flat+67)%(prec-1)+1
        weight_seed_neg = np.arange(37, weight_split_size_flat+37)%(prec-1)+1
        rand_input = torch.from_numpy(input_seed).reshape(input_split_size).to(device)
        rand_weight_pos = torch.from_numpy(weight_seed_pos).reshape(weight_split_size).to(device)
        rand_weight_neg = torch.from_numpy(weight_seed_neg).reshape(weight_split_size).to(device)
    elif global_usesync:
        seed = torch.from_numpy(np.random.permutation(prec)).to(device)
        input_pos = torch.randint(prec, input_split_size, dtype=torch.int64, device=device)
        weight_pos_pos = torch.randint(prec, weight_split_size, dtype=torch.int64, device=device)
        weight_neg_pos = torch.randint(prec, weight_split_size, dtype=torch.int64, device=device)
    else:
        rand_input = torch.randint(prec, input_split_size, dtype=compare_type, device=device)
        rand_weight_pos = torch.randint(prec, weight_split_size, dtype=compare_type, device=device)
        rand_weight_neg = torch.randint(prec, weight_split_size, dtype=compare_type, device=device)
    
    result_pos = []
    result_neg = []
    
    pool_cnt = pool[0]*pool[1]
    stride_pool = list(stride[:])
    stride_pool[0] *= pool[0]
    stride_pool[1] *= pool[1]
            
    for k in np.arange(0, bit_length, pool_cnt):
        result_pos.append([])
        result_neg.append([])
        if global_lfsr:
            rand_input = ((rand_input//32)%2+rand_input//64)%2+2*(rand_input%64)
            rand_weight_pos = ((rand_weight_pos//32)%2+rand_weight_pos//64)%2+2*(rand_weight_pos%64)
            rand_weight_neg = ((rand_weight_neg//32)%2+rand_weight_neg//64)%2+2*(rand_weight_neg%64)
        elif global_usesync:
            rand_input = seed[input_pos]
            rand_weight_pos = seed[weight_pos_pos]
            rand_weight_neg = seed[weight_neg_pos]
            input_pos = (input_pos+1)%prec
            weight_pos_pos = (weight_pos_pos+1)%prec
            weight_neg_pos = (weight_neg_pos+1)%prec
        else:
            rand_input.random_(0,prec)
            rand_weight_pos.random_(0,prec)
            rand_weight_neg.random_(0,prec)
        input_bit = (input_split > rand_input).to(compute_type)
        w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
        w_neg_bit = (w_neg_split > rand_weight_pos).to(compute_type)
        for i in range(pool[0]):
            for j in range(pool[1]):
                input_bit_cur = input_bit[...,i:,j:]
                result_pos[-1].append(F.conv2d(input_bit_cur, w_pos_bit, stride=stride_pool).sign())
                result_neg[-1].append(F.conv2d(input_bit_cur, w_neg_bit, stride=stride_pool).sign())
        result_pos[-1] = torch.stack(result_pos[-1], 0)
        result_neg[-1] = torch.stack(result_neg[-1], 0)
    result_pos = torch.stack(result_pos, 0)
    result_neg = torch.stack(result_neg, 0)
    
    return result_pos, result_neg

def conv2d_or_stream_out_pool(activation, weight, bit_length, padding, stride, pool):
    if global_mixadd:
        result_pos, result_neg = conv2d_or_stream_out_pool_prep_mixed(activation, weight, bit_length, padding, stride, pool)
    else:
        result_pos, result_neg = conv2d_or_stream_out_pool_prep(activation, weight, bit_length, padding, stride, pool)
    
    w_pos = weight.clamp(0,100)
    w_neg = -(weight.clamp(-100,0))
    activation = activation.to(w_pos.dtype)
    
    if global_mixadd:
        activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
        i_x = activation.size(2)
        i_y = activation.size(3)
        f_x = weight.size(2)
        f_y = weight.size(3)
        result_pos_value = []
        result_neg_value = []
        for i in range(f_x):
            for j in range(f_y):
                a_sec = activation[...,i:i_x-f_x+i+1,j:i_y-f_y+j+1].clone()
                w_pos_sec = w_pos[...,i:i+1,j:j+1].clone()
                w_neg_sec = w_pos[...,i:i+1,j:j+1].clone()
                result_pos_value.append(1-torch.exp(-F.conv2d(a_sec, w_pos_sec, stride=stride)))
                result_neg_value.append(1-torch.exp(-F.conv2d(a_sec, w_neg_sec, stride=stride)))
        result_pos_value = torch.stack(result_pos_value, 0).sum(0)
        result_neg_value = torch.stack(result_neg_value, 0).sum(0)
    else:
        result_pos_value = 1-torch.exp(-F.conv2d(activation, w_pos, padding=padding, stride=stride))
        result_neg_value = 1-torch.exp(-F.conv2d(activation, w_neg, padding=padding, stride=stride))

    result, result_value = subtract_relu_pool((result_pos, result_pos_value), (result_neg, result_neg_value), pool)
    return result, result_value
    
def conv2d_or_stream_out(activation, weight, bit_length, padding, stride):
    prec = global_prec
    bit_range = prec - 1

    device = activation.device
    
    input_split = (activation.data*prec).to(compare_type)
    w_pos_split = (weight.data*prec).clamp(0,bit_range).to(compare_type)
    w_neg_split = -((weight.data*prec).clamp(-bit_range,0)).to(compare_type)
#     print(input_split)
    
    if global_share:
        input_split_size = input_split.size()[-3:]
        weight_split_size= w_pos_split.size()[-3:]
    else:
        input_split_size = input_split.size()
        weight_split_size= w_pos_split.size()
    rand_input = torch.randint(prec, input_split_size, dtype=compare_type, device=device)
    rand_weight_pos = torch.randint(prec, weight_split_size, dtype=compare_type, device=device)
    rand_weight_neg = torch.randint(prec, weight_split_size, dtype=compare_type, device=device)
    
    result_pos = []
    result_neg = []
    input_bit = (input_split > rand_input).to(compute_type)
    w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
    w_neg_bit = (w_neg_split > rand_weight_neg).to(compute_type)
    result_pos.append(F.conv2d(input_bit, w_pos_bit, padding=padding, stride=stride).sign())
    result_neg.append(F.conv2d(input_bit, w_neg_bit, padding=padding, stride=stride).sign())
    
    for j in range(1, bit_length):
        rand_input.random_(0, prec)
        rand_weight_pos.random_(0, prec)
        rand_weight_neg.random_(0, prec)
        input_bit = (input_split > rand_input).to(compute_type)
        w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
        w_neg_bit = (w_neg_split > rand_weight_neg).to(compute_type)
        result_pos.append(F.conv2d(input_bit, w_pos_bit, padding=padding, stride=stride).sign())
        result_neg.append(F.conv2d(input_bit, w_neg_bit, padding=padding, stride=stride).sign())

    result_pos = torch.stack(result_pos, 0)
    result_neg = torch.stack(result_neg, 0)

    w_pos = weight.clamp(0,100)
    w_neg = -(weight.clamp(-100,0))
    activation = activation.to(w_pos.dtype)
    result_pos_value = 1-torch.exp(-F.conv2d(activation, w_pos, padding=padding, stride=stride))
    result_pos_value.data = result_pos.mean(0)
    result_neg_value = 1-torch.exp(-F.conv2d(activation, w_neg, padding=padding, stride=stride))
    result_neg_value.data = result_neg.mean(0)
    
    return (result_pos, result_pos_value), (result_neg, result_neg_value)

def conv2d_or_stream_pool(activation, weight, padding, stride, pool):
    '''
    Pooled input followed by pooling. What if it's not followed by pooling?
    '''
    prec = global_prec
    activation, a_v = activation
    bit_length = activation.size(0)*activation.size(1)
    bit_range = prec-1
    device = activation.device
    
    # Prescale inputs for bit stream generation
    w_pos_split = (weight.data*prec).clamp(0,bit_range).to(compare_type)
    w_neg_split = -((weight.data*prec).clamp(-bit_range,0)).to(compare_type)
    
    if global_share:
        weight_split_size= w_pos_split.size()[-3:]
    else:
        weight_split_size= w_pos_split.size()

    # Currently generation is shared? This can change (relatively easily) by moving it into the computation loop)
    if global_lfsr:
        weight_split_size_flat = np.array(weight_split_size).prod()
        weight_seed_pos = np.arange(67, weight_split_size_flat+67)%(prec-1)+1
        weight_seed_neg = np.arange(37, weight_split_size_flat+37)%(prec-1)+1
        rand_weight_pos = torch.from_numpy(weight_seed_pos).reshape(weight_split_size).to(device)
        rand_weight_neg = torch.from_numpy(weight_seed_neg).reshape(weight_split_size).to(device)
    elif global_usesync:
        seed = torch.from_numpy(np.random.permutation(prec)).to(device)
        weight_pos_pos = torch.randint(prec, weight_split_size, dtype=torch.int64, device=device)
        weight_neg_pos = torch.randint(prec, weight_split_size, dtype=torch.int64, device=device)
    else:
        rand_weight_pos = torch.randint(prec, weight_split_size, dtype=compare_type, device=device)
        rand_weight_neg = torch.randint(prec, weight_split_size, dtype=compare_type, device=device)
    
    result_pos = []
    result_neg = []
    
    pool_cnt = pool[0]*pool[1]
    input_split = F.pad(activation, (pool[0], pool[0], pool[1], pool[1]))
    
    stride_pool = list(stride[:])
    stride_pool[0] *= pool[0]
    stride_pool[1] *= pool[1]
    
    i_x = input_split.size(-2)
    i_y = input_split.size(-1)
    f_x = weight.size(2)
    f_y = weight.size(3)
    
            
    for k in np.arange(0, bit_length, pool_cnt):
        result_pos.append([])
        result_neg.append([])
        if global_lfsr:
            rand_weight_pos = ((rand_weight_pos//32)%2+rand_weight_pos//64)%2+2*(rand_weight_pos%64)
            rand_weight_neg = ((rand_weight_neg//32)%2+rand_weight_neg//64)%2+2*(rand_weight_neg%64)
        elif global_usesync:
            rand_weight_pos = seed[weight_pos_pos]
            rand_weight_neg = seed[weight_neg_pos]
            weight_pos_pos = (weight_pos_pos+1)%prec
            weight_neg_pos = (weight_neg_pos+1)%prec
        else:
            rand_weight_pos.random_(0,prec)
            rand_weight_neg.random_(0,prec)
        w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
        w_neg_bit = (w_neg_split > rand_weight_pos).to(compute_type)
        cnt = 0
        for i in range(pool[0]):
            for j in range(pool[1]):
                input_bit_cur = input_split[k//pool_cnt,cnt,...,i:,j:]
                if global_mixadd:
                    result_pos_cur = []
                    result_neg_cur = []
                    for l in range(f_x):
                        for m in range(f_y):
                            input_bit_cur_cur = input_bit_cur[...,l:i_x-f_x+l+1,m:i_y-f_y+m+1]
                            result_pos_cur.append(F.conv2d(input_bit_cur_cur, w_pos_bit[...,l:l+1,m:m+1], stride=stride_pool).sign())
                            result_neg_cur.append(F.conv2d(input_bit_cur_cur, w_neg_bit[...,l:l+1,m:m+1], stride=stride_pool).sign())
                    result_pos_cur = torch.stack(result_pos_cur, 0)
                    result_neg_cur = torch.stack(result_neg_cur, 0)
                    result_pos[-1].append(result_pos_cur.sum(0))
                    result_neg[-1].append(result_neg_cur.sum(0))
                else:
                    result_pos[-1].append(F.conv2d(input_bit_cur, w_pos_bit, stride=stride_pool).sign())
                    result_neg[-1].append(F.conv2d(input_bit_cur, w_neg_bit, stride=stride_pool).sign())
                cnt += 1
        result_pos[-1] = torch.stack(result_pos[-1], 0)
        result_neg[-1] = torch.stack(result_neg[-1], 0)
    result_pos = torch.stack(result_pos, 0)
    result_neg = torch.stack(result_neg, 0)
    
    w_pos = weight.clamp(0,100)
    w_neg = -(weight.clamp(-100,0))
    activation = activation.to(w_pos.dtype)
    
    if global_mixadd:
        a_v = F.pad(a_v, (padding[0], padding[0], padding[1], padding[1]))
        result_pos_value = []
        result_neg_value = []
        for i in range(f_x):
            for j in range(f_y):
                a_sec = a_v[...,i:i_x-f_x+i+1,j:i_y-f_y+j+1].clone()
                w_pos_sec = w_pos[...,i:i+1,j:j+1].clone()
                w_neg_sec = w_pos[...,i:i+1,j:j+1].clone()
                result_pos_value.append(1-torch.exp(-F.conv2d(a_sec, w_pos_sec, stride=stride)))
                result_neg_value.append(1-torch.exp(-F.conv2d(a_sec, w_neg_sec, stride=stride)))
        result_pos_value = torch.stack(result_pos_value, 0).sum(0)
        result_neg_value = torch.stack(result_neg_value, 0).sum(0)
    else:
        result_pos_value = 1-torch.exp(-F.conv2d(a_v, w_pos, padding=padding, stride=stride))
        result_neg_value = 1-torch.exp(-F.conv2d(a_v, w_neg, padding=padding, stride=stride))
    result, result_value = subtract_relu_pool((result_pos, result_pos_value), (result_neg, result_neg_value), pool)
    return result, result_value

def conv2d_or_stream(activation, weight, padding, stride):
    a, a_v = activation
    w = weight
    device = a.device
    bit_length = a.size(0)
    bit_range = bit_length-1
    
    w_pos = (weight.data*bit_length).clamp(0,bit_range).to(compare_type)
    w_neg = -(weight.data*bit_length).clamp(-bit_range,0).to(compare_type)
    
    if global_share:
        w_size = w.size(-1)
        rand_weight_pos = torch.randint(bit_length, [w_size], dtype=compare_type, device=device)
        rand_weight_neg = torch.randint(bit_length, [w_size], dtype=compare_type, device=device)
    else:
        w_size = w.size()
        rand_weight_pos = torch.randint(bit_length, w_size, dtype=compare_type, device=device)
        rand_weight_neg = torch.randint(bit_length, w_size, dtype=compare_type, device=device)
        
    result_pos = []
    result_neg = []
    w_pos_bit = (w_pos > rand_weight_pos).to(compute_type)
    w_neg_bit = (w_neg > rand_weight_neg).to(compute_type)
    result_pos.append(F.conv2d(a[0], w_pos_bit, padding=padding, stride=stride).sign())
    result_neg.append(F.conv2d(a[0], w_neg_bit, padding=padding, stride=stride).sign())
    
    for j in range(1, bit_length):
        rand_weight_pos.random_(0, bit_length)
        rand_weight_neg.random_(0, bit_length)
        w_pos_bit = (w_pos > rand_weight_pos).to(compute_type)
        w_neg_bit = (w_neg > rand_weight_neg).to(compute_type)
        result_pos.append(F.conv2d(a[j], w_pos_bit, padding=padding, stride=stride).sign())
        result_neg.append(F.conv2d(a[j], w_neg_bit, padding=padding, stride=stride).sign())
        
    result_pos = torch.stack(result_pos, 0)
    result_neg = torch.stack(result_neg, 0)
        
    w_pos = weight.clamp(0,100)
    w_neg = -(weight.clamp(-100,0))
    a_v = a_v.to(w_pos.dtype)
    result_pos_value = 1-torch.exp(-F.conv2d(a_v, w_pos, padding=padding, stride=stride))
    result_pos_value.data = result_pos.mean(0)
    result_neg_value = 1-torch.exp(-F.conv2d(a_v, w_neg, padding=padding, stride=stride))
    result_neg_value.data = result_neg.mean(0)
    
    return (result_pos, result_pos_value), (result_neg, result_neg_value)

def conv2d_or_shared(activation, weight, bit_length, padding, stride):
    bit_range = bit_length - 1

    device = activation.device
    
    input_split = (activation*bit_length).to(compare_type)
    w_pos_split = (weight*bit_length).clamp(0,bit_range).to(compare_type)
    w_neg_split = -((weight*bit_length).clamp(-bit_range,0)).to(compare_type)
#     print(input_split)
    
    if global_share:
        input_split_size = input_split.size()[-3:]
        weight_split_size= w_pos_split.size()[-3:]
    else:
        input_split_size = input_split.size()
        weight_split_size= w_pos_split.size()
    rand_input = torch.randint(bit_length, input_split_size, dtype=compare_type, device=device)
    rand_weight_pos = torch.randint(bit_length, weight_split_size, dtype=compare_type, device=device)
    rand_weight_neg = torch.randint(bit_length, weight_split_size, dtype=compare_type, device=device)
    
    input_bit = (input_split > rand_input).to(compute_type)
    w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
    w_neg_bit = (w_neg_split > rand_weight_neg).to(compute_type)
    result_pos = F.conv2d(input_bit, w_pos_bit, padding=padding, stride=stride).sign()
    result_neg = F.conv2d(input_bit, w_neg_bit, padding=padding, stride=stride).sign()
    
    for j in range(1, bit_length):
        rand_input.random_(0, bit_length)
        rand_weight_pos.random_(0, bit_length)
        rand_weight_neg.random_(0, bit_length)
        input_bit = (input_split > rand_input).to(compute_type)
        w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
        w_neg_bit = (w_neg_split > rand_weight_neg).to(compute_type)
#         print(input_bit)
        result_pos += F.conv2d(input_bit, w_pos_bit, padding=padding, stride=stride).sign()
        result_neg += F.conv2d(input_bit, w_neg_bit, padding=padding, stride=stride).sign()
#         print(result_pos)
            
    result_pos_scale = result_pos.to(torch.float32) / bit_length
    result_neg_scale = result_neg.to(torch.float32) / bit_length
    return result_pos_scale, result_neg_scale

def conv2d_or_split(activation, weight, padding, stride, output_pos, output_neg, err=7, split=4):
    bit_length = 2**err
    bit_range = bit_length - 1
    
    split_size = activation.size(1)/split
    device = activation.device
    
    input_split = torch.split((activation*bit_length).to(compare_type), split_size, 1)
    w_pos_split = torch.split((weight*bit_length).clamp(0,bit_range).to(compare_type), split_size, 1)
    w_neg_split = torch.split(-(weight*bit_length).clamp(-bit_range,0).to(compare_type), split_size, 1)
    
    input_split_size = input_split[0].size()
    weight_split_size= w_pos_split[0].size()
    rand_input = torch.randint(bit_length, input_split_size, dtype=compare_type, device=device)
    rand_weight_pos = torch.randint(bit_length, weight_split_size, dtype=compare_type, device=device)
    rand_weight_neg = torch.randint(bit_length, weight_split_size, dtype=compare_type, device=device)
    
    input_bit = (input_split[0] > rand_input).to(compute_type)
    w_pos_bit = (w_pos_split[0] > rand_weight_pos).to(compute_type)
    w_neg_bit = (w_neg_split[0] > rand_weight_neg).to(compute_type)
    result_pos = F.conv2d(input_bit, w_pos_bit, padding=padding, stride=stride).sign()
    result_neg = F.conv2d(input_bit, w_neg_bit, padding=padding, stride=stride).sign()
    
    for i in range(1, split):
        rand_input.random_(0, bit_length).round_()
        rand_weight_pos.random(0, bit_length).round_()
        rand_weight_neg.random(0, bit_length).round_()
        input_bit = (input_split[i] > rand_input).to(compute_type)
        weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(compute_type)
        weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(compute_type)
        result_pos += F.conv2d(input_bit, w_pos_bit, padding=padding, stride=stride).sign()
        result_neg += F.conv2d(input_bit, w_neg_bit, padding=padding, stride=stride).sign()
    
    for j in range(1, bit_length):
        for i in range(split):
            rand_input.random_(0, bit_range).round_()
            rand_weight_pos.random(0, bit_range).round_()
            rand_weight_neg.random(0, bit_range).round_()
            input_bit = (input_split[i] > rand_input).to(compute_type)
            weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(compute_type)
            weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(compute_type)
            result_pos += F.conv2d(input_bit, w_pos_bit, padding=padding, stride=stride).sign()
            result_neg += F.conv2d(input_bit, w_neg_bit, padding=padding, stride=stride).sign()
            
    result_pos_scale = result_pos / bit_length
    result_neg_scale = result_neg / bit_length
    return result_pos_scale, result_neg_scale

def conv2d_or_acc(activation, weight, padding, stride, output_pos, output_neg, err=7, sync=global_sync):
    bit_length = 2**err
    sec_length = 2**sync
    
    bit_range = bit_length - 1
    
    a_scale = (activation*bit_length).to(compute_type)
    w_pos_scale = (weight*bit_length).clamp(0,bit_range).to(compute_type)
    w_neg_scale = -(weight*bit_length).clamp(-bit_range,0).to(compute_type)
    
    seed = torch.from_numpy(np.tile(np.random.permutation(sec_length), 2)).to(activation.device, compute_type)
    scale = torch.from_numpy(2**np.arange(0,err,sync)).to(activation.device, compute_type)
    
    a_size = list(activation.size())
    w_size = list(weight.size())
    
    a_size.append(err//sync)
    w_size.append(err//sync)
    
    a_size = tuple(a_size)
    w_size = tuple(w_size)
    
    for i in range(bit_length//sec_length):
        pos_a = torch.randint(0, sec_length, a_size, device=activation.device)
        pos_w_pos = torch.randint(0, sec_length, w_size, device=activation.device)
        pos_w_neg = torch.randint(0, sec_length, w_size, device=activation.device)
        for j in range(sec_length):
            val_a = torch.sum(seed[pos_a] * scale, -1)#.to(compute_type)
            val_w_pos = torch.sum(seed[pos_w_pos] * scale, -1)#.to(compute_type)
            val_w_neg = torch.sum(seed[pos_w_neg] * scale, -1)#.to(compute_type)
            
            a_bit = (a_scale > val_a).to(compute_type)
#             print(a_bit.dtype)
#             print(a_bit)
            w_pos_bit = (w_pos_scale > val_w_pos).to(compute_type)
            w_neg_bit = (w_neg_scale > val_w_neg).to(compute_type)

            output_pos += (F.conv2d(a_bit, w_pos_bit, padding=padding, stride=stride).sign())#.to(torch.float32)#.sign()
            output_neg += (F.conv2d(a_bit, w_neg_bit, padding=padding, stride=stride).sign())#.to(torch.float32)#.sign()
            
            pos_a = (pos_a+1)#%sec_length
            pos_w_pos = (pos_w_pos+1)#%sec_length
            pos_w_neg = (pos_w_neg+1)#%sec_length
            
    output_pos /= bit_length
    output_neg /= bit_length
    return torch.stack((output_pos, output_neg))

def conv2d_or_subtract(activation, weight, padding=0, stride=1, add_or=False, convert=False, err=16, add_full=1, true_or=False, add_count=False, noerr=False or global_noerr):
    '''
    image (B, Cin, Hin, Win)
    weight (B, Cout, Hout, Wout)
    There are two choices:
        1. Either I can do an img2col operation, treat it as an FC layer
        2. Or I do the convolution and handle all the complexities
    '''
    err_mult = 2**err
    weight_pos = weight.clamp(0,100)
    weight_neg = -weight.clamp(-100,0)

    activation_pos_cor = F.conv2d(activation, weight_pos, padding=padding, stride=stride)
    activation_neg_cor = F.conv2d(activation, weight_neg, padding=padding, stride=stride)

    if add_count:
        pass
    elif true_or:
        activation_unf = F.unfold(activation, weight.size(2), padding=padding, stride=stride)
        weight_pos_unf = weight_pos.reshape(weight.size(0),-1)
        weight_neg_unf = weight_neg.reshape(weight.size(0),-1)
        res_unf_pos, res_unf_neg = linear_or_reduce(activation_unf, weight_pos_unf, weight_neg_unf)
        activation_pos = F.fold(res_unf_pos, (activation_pos_cor.size(2), activation_pos_cor.size(3)), (1,1))
        activation_neg = F.fold(res_unf_neg, (activation_neg_cor.size(2), activation_neg_cor.size(3)), (1,1))
    else:
#         activation_pos_exp = (1-torch.exp(-activation_pos_cor))
#         activation_neg_exp = (1-torch.exp(-activation_neg_cor))
#         activation_pos_tanh = torch.tanh(activation_pos_cor)
#         activation_neg_tanh = torch.tanh(activation_neg_cor)
#         activation_pos = coef*activation_pos_exp + (1-coef)*activation_pos_tanh
#         activation_neg = coef*activation_neg_exp + (1-coef)*activation_neg_tanh
        activation_pos = (1-torch.exp(-activation_pos_cor))
        activation_neg = (1-torch.exp(-activation_neg_cor))
    if noerr:
        return activation_pos - activation_neg
    
    if add_count:
        print("Here")
        output = activation_pos - activation_neg
        output_data = sc_extension.conv2d_count(activation.data, weight.data, err_mult, padding, stride)
        output.data = output_data
    elif add_or:
        output = activation_pos - activation_neg
        with torch.no_grad():
            if use_or:
#                 output_data = sc_extension.conv2d_or(activation.data, weight.data, err_mult, padding, stride, add_full).to(output.dtype)
                output_data = conv2d_or_shared(activation.data, weight.data, err_mult, padding, stride)
            elif not use_sync:
                output_data = sc_extension.conv2d_acc(activation.data, weight.data, err_mult, padding, stride, add_full)
            else:
                output_pos = torch.zeros_like(output.data)
                output_neg = torch.zeros_like(output.data)
                output_data = conv2d_or_acc(activation.data, weight.data, padding, stride, output_pos, output_neg, err)
        output.data = (output_data[0] - output_data[1]).to(output.dtype)
    else:
#         err_mult = err_mult/2
        activation_pos_scale = activation_pos.data
        activation_neg_scale = activation_neg.data
        error_pos = torch.randn_like(activation_pos_scale) * torch.sqrt(activation_pos_scale*(1-activation_pos_scale)/err_mult)
        error_neg = torch.randn_like(activation_neg_scale) * torch.sqrt(activation_neg_scale*(1-activation_neg_scale)/err_mult)
        
        activation_pos.data = activation_pos.data + error_pos
        activation_neg.data = activation_neg.data + error_neg
        output = activation_pos - activation_neg
    return output

def linear_add(activation, weight, bit_length, split_size=None):
    prec = global_prec
    bit_range = prec-1
    if split_size is None:
        split_size = weight.size(0)
    
    device = activation.device
    
    input_split = (activation.data*prec).to(compare_type)
    w_pos_split = (weight.data*prec).clamp(0,bit_range).to(compare_type)
    w_neg_split = -((weight.data*prec).clamp(-bit_range,0).to(compare_type))
    
    if global_share:
        input_split_size = input_split[0].size(-1)
        weight_split_size= w_pos_split[0].size(-1)
        rand_input = torch.randint(prec, [input_split_size], dtype=compare_type, device=device)
        rand_weight_pos = torch.randint(prec, [weight_split_size], dtype=compare_type, device=device)
        rand_weight_neg = torch.randint(prec, [weight_split_size], dtype=compare_type, device=device)
    else:
        input_split_size = input_split[0].size()
        weight_split_size= w_pos_split[0].size()
        rand_input = torch.randint(prec, input_split_size, dtype=compare_type, device=device)
        rand_weight_pos = torch.randint(prec, weight_split_size, dtype=compare_type, device=device)
        rand_weight_neg = torch.randint(prec, weight_split_size, dtype=compare_type, device=device)
    
    result_pos = []
    result_neg = []
    
    split_interval = weight.size(0) // split_size
    bit_length = bit_length // split_size
    
    for j in range(0, bit_length):
        result_pos.append([])
        result_neg.append([])
        for k in range(split_size):
            rand_input.random_(0,prec)
            rand_weight_pos.random_(0,prec)
            rand_weight_neg.random_(0,prec)
            input_bit = (input_split > rand_input).to(compute_type)
            w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
            w_neg_bit = (w_neg_split > rand_weight_neg).to(compute_type)
            result_pos[-1].append(F.linear(input_bit, w_pos_bit[k*split_interval:(k+1)*split_interval]))
            result_neg[-1].append(F.linear(input_bit, w_neg_bit[k*split_interval:(k+1)*split_interval]))
        result_pos[-1] = torch.cat(result_pos[-1], 1)
        result_neg[-1] = torch.cat(result_neg[-1], 1)
    result_pos = torch.stack(result_pos, 0)
    result_neg = torch.stack(result_neg, 0)
    
    result_value = F.linear(activation, weight)
    result_pos_scale = result_pos.mean(0).to(torch.float32)
    result_neg_scale = result_neg.mean(0).to(torch.float32)
    result = result_pos_scale - result_neg_scale
    result_value.data = result
    return result_value

def conv2d_add(activation, weight, bit_length, padding, stride, split_size=None):
    prec = global_prec
    bit_range = prec-1

    if split_size is None:
        split_size = global_split
    device = activation.device
    
    input_split = (activation.data*prec).to(compare_type)
    w_pos_split = (weight.data*prec).clamp(0,bit_range).to(compare_type)
    w_neg_split = -((weight.data*prec).clamp(-bit_range,0)).to(compare_type)
    
    input_split = F.pad(input_split, (padding[0], padding[0], padding[1], padding[1]))
    
    if global_share:
        input_split_size = input_split.size()[-3:]
        weight_split_size= w_pos_split.size()[-3:]
    else:
        input_split_size = input_split.size()
        weight_split_size= w_pos_split.size()
    rand_input = torch.randint(prec, input_split_size, dtype=compare_type, device=device)
    rand_weight_pos = torch.randint(prec, weight_split_size, dtype=compare_type, device=device)
    rand_weight_neg = torch.randint(prec, weight_split_size, dtype=compare_type, device=device)
    
    result_pos = []
    result_neg = []
    
    i_x = input_split.size(-2)
    i_y = input_split.size(-1)
    f_x = weight.size(2)
    f_y = weight.size(3)
    
    split_1d = int(np.sqrt(split_size))
    stride_split = [stride[0]*split_1d,stride[1]*split_1d]
    bit_length = bit_length // split_size
    for j in range(0, bit_length):
        result_pos.append([])
        result_neg.append([])
        for k in range(split_1d):
            for l in range(split_1d):
                rand_input.random_(0, prec)
                rand_weight_pos.random_(0, prec)
                rand_weight_neg.random_(0, prec)
                input_bit = (input_split > rand_input).to(compute_type)
                w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
                w_neg_bit = (w_neg_split > rand_weight_neg).to(compute_type)
        #         print(input_bit)
                if global_mixadd:
                    result_pos_cur = []
                    result_neg_cur = []
                    for m in range(f_x):
                        for n in range(f_y):
                            input_bit_cur = input_bit[...,m:i_x-f_x+m+1,n:i_y-f_y+n+1]
                            result_pos_cur.append(F.conv2d(input_bit_cur, w_pos_bit[...,m:m+1,n:n+1], stride=stride_split))
                            result_neg_cur.append(F.conv2d(input_bit_cur, w_neg_bit[...,m:m+1,n:n+1], stride=stride_split))
                    result_pos_cur = torch.stack(result_pos_cur, 0)
                    result_neg_cur = torch.stack(result_neg_cur, 0)
                    result_pos[-1].append(result_pos_cur.sum(0))
                    result_neg[-1].append(result_neg_cur.sum(0))
                else:
                    result_pos[-1].append(F.conv2d(input_bit[...,k:,l:], w_pos_bit, stride=stride_split))
                    result_neg[-1].append(F.conv2d(input_bit[...,k:,l:], w_neg_bit, stride=stride_split))

        result_pos[-1] = torch.stack(result_pos[-1],0)
        result_neg[-1] = torch.stack(result_neg[-1],0)
    result_pos = torch.stack(result_pos,0)
    result_neg = torch.stack(result_neg,0)
    
    result_pos = unpool(result_pos)
    result_neg = unpool(result_neg)
    
    if global_mixadd:
        w_pos = weight.clamp(0,100)
        w_neg = weight.clamp(-100,0)
        activation = F.pad(activation, (padding[0], padding[1], padding[2], padding[3]))
        result_pos_value = []
        result_neg_value = []
        for i in range(f_x):
            for j in range(f_y):
                a_sec = a_v[...,i:i_x-f_x+i+1,j:j_y-f_y+j+1].clone()
                w_pos_sec = w_pos[...,i:i+1,j:j+1].clone()
                w_neg_sec = w_neg[...,i:i+1,j:j+1].clone()
                result_pos_value.append(1-torch.exp(-F.conv2d(a_sec, w_pos_sec, stride=stride)))
                result_neg_value.append(1-torch.exp(-F.conv2d(a_sec, w_neg_sec, stride=stride)))
        result_pos_value = torch.stack(result_pos_value, 0)
        result_neg_value = torch.stack(result_neg_value, 0)
        result_value = result_pos_value.sum(0)-result_neg_value.sum(0)
    else:
        result_value = F.conv2d(activation, weight, padding=padding, stride=stride)
    result_pos_scale = result_pos.mean(0).to(torch.float32)
    result_neg_scale = result_neg.mean(0).to(torch.float32)
    result = result_pos_scale - result_neg_scale
    result_value.data = result
#     print(result.size(), result_value.size(), (result_value/(result+1e-5)).mean())
    return result_value
    

def quantize(input, quant=False, prec=8):
    prec_2 = 2**prec
    if quant:
        input = (input * prec_2).round().clamp(-prec_2+1, prec_2-1)/prec_2
    return input

            
class QuantizeConv2d(nn.Conv2d):
    '''
    Quantized conv2d with mask for pruning
    '''
    def __init__(self, thres=0, prec=8, *kargs, **kwargs):
        super(QuantizeConv2d, self).__init__(*kargs, **kwargs)
        self.thres = thres
        self.prec = prec
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('weight_org', self.weight.data.clone())
    
    def forward(self, input, quant=False, pruned=False, mult=None):
        # If mult exists, overwrites pruning
        if mult is not None:
            self.thres = mult
        input.data, _ = quantize(input.data, quant=quant, prec=self.prec)
        self.weight.data, _=quantize(self.weight_org, mask=self.mask, quant=quant, pruned=pruned, mult=self.thres, prec=self.prec)
        out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        return out
    
class QuantizeLinear(nn.Linear):
    def __init__(self, thres=0, prec=8, *kargs, **kwargs):
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)
        self.thres = thres
        self.prec = prec
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('weight_org', self.weight.data.clone())
    
    def forward(self, input, quant=False, pruned=False, mult=None):
        # If mult exists, overwrites pruning
        if mult is not None:
            self.thres = mult
        input.data, _ = quantize(input.data, quant=quant, prec=self.prec)
        self.weight.data, _=quantize(self.weight_org, mask=self.mask, quant=quant, pruned=pruned, mult=self.thres, prec=self.prec)
        out = nn.functional.linear(input, self.weight)
        return out
    
class Conv2d_Or_Streamout_relu_pool(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        try:
            self.pool = kwargs['pool']
        except:
            self.pool = (2,2)
        else:
            del(kwargs['pool'])
        super(Conv2d_Or_Streamout_relu_pool, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    def forward(self, input, prec=7, err=7):
        self.weight.data = quantize(self.weight_org, quant=True, prec=prec)
        out = conv2d_or_stream_out_pool(input, self.weight, bit_length=2**err, padding=self.padding, stride=self.stride, pool=self.pool)
        return out
    
class Conv2d_Or_Streamout(nn.Conv2d):
    '''
    Or-based conv2d without conversion
    '''
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Or_Streamout, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    def forward(self, input, prec=7, err=7):
        self.weight.data = quantize(self.weight_org, quant=True, prec=prec)
        out = conv2d_or_stream_out(input, self.weight, bit_length=2**err, padding=self.padding, stride=self.stride)
        return out
    
class Conv2d_Or_Stream_relu_pool(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        try:
            self.pool = kwargs['pool']
        except:
            self.pool = (2,2)
        else:
            del(kwargs['pool'])
        super(Conv2d_Or_Stream_relu_pool, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    def forward(self, input):
        self.weight.data = quantize(self.weight_org, quant=True, prec=7)
        out = conv2d_or_stream_pool(input, self.weight, padding=self.padding, stride=self.stride, pool=self.pool)
        return out
    
class Conv2d_Or_Stream(nn.Conv2d):
    '''
    Or-based conv2d without conversion and using stream input
    '''
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Or_Stream, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    def forward(self, input):
        self.weight.data = quantize(self.weight_org, quant=True, prec=7)
        out = conv2d_or_stream(input, self.weight, padding=self.padding, stride=self.stride)
        return out
    
class Linear_Add(nn.Linear):
    '''
    SC Linear using binary add
    '''
    def __init__(self, *kargs, **kwargs):
        super(Linear_Add, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    def forward(self, input, add_or=False, prec=7, err=7):
        input.data = quantize(input.data, quant=True, prec=7)
        self.weight.data = quantize(self.weight_org, quant=True, prec=7)
        out = linear_add(input, self.weight, bit_length=2**err)
        return out
    
class Conv2d_Add(nn.Conv2d):
    '''
    SC Conv2d using binary add. Using this will not be better than bisc-mvm. But try this anyways.
    '''
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Add, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
        
    def forward(self, input, add_or=False, prec=7, err=7):
        input.data = quantize(input.data, quant=True, prec=7)
        self.weight.data = quantize(self.weight_org, quant=True, prec=7)
        out = conv2d_add(input, self.weight, bit_length=2**err, padding=self.padding, stride=self.stride)
        return out
    
class Conv2d_Or(nn.Conv2d):
    '''
    Quantized conv2d with mask for pruning
    '''
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Or, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    
    def forward(self, input, add_or=False, prec=7, err=7, add_full=1, true_or=False, add_count=False):
        # If mult exists, overwrites pruning
        add_or = self.add_or or add_or
        if prec is not None:
            quant=True
        else:
            quant=False
            prec=8
#         if input.size(1)!=3:
        input.data = quantize(input.data, quant=quant, prec=prec)
        self.weight.data =quantize(self.weight_org, quant=quant, prec=prec)
        out = conv2d_or_subtract(input, self.weight, padding=self.padding, stride=self.stride, add_or=add_or, err=err, add_full=add_full, true_or=true_or, add_count=add_count)
        return out
    
class Conv2d_Or_Add(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Or_Add, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        
    def forward(self, input_pos, input_neg, prec=None):
        if prec is not None:
            quant=True
        else:
            quant=False
            prec=7
        input_pos.data = quantize(input_pos.data, quant=quant, prec=prec)
        input_neg.data = quantize(input_neg.data, quant=quant, prec=prec)
        self.weight.data = quantize(self.weight_org, quant=quant, prec=prec)
        out_pos, out_neg = conv2d_or_add(input_pos, input_neg, self.weight, padding=self.padding, stride=self.stride)
        return out_pos, out_neg

class Linear_Or_Stream(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(Linear_Or_Stream, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    def forward(self, input):
        self.weight.data = quantize(self.weight_org, quant=True, prec=7)
        out = linear_or_stream(input, self.weight)
        return out

class Linear_Or(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(Linear_Or, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    
    def forward(self, input, add_or=False, prec=7, err=7, add_full=1, true_or=False, add_count=False):
        if prec is not None:
            quant=True
        else:
            quant=False
            prec=8
        add_or = self.add_or or add_or
        input.data = quantize(input.data, quant=quant, prec=prec)
        self.weight.data =quantize(self.weight_org, quant=quant, prec=prec)
        out =linear_or_subtract(input, self.weight, add_or=add_or,  err=err, add_full=add_full, true_or=true_or, add_count=add_count)
        return out
    
class Linear_Or_Add(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(Linear_Or_Add, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        
    def forward(self, input_pos, input_neg, prec=None):
        if prec is not None:
            quant=True
        else:
            quant=False
            prec=7
        input_pos.data = quantize(input_pos.data, quant=quant, prec=prec)
        input_neg.data = quantize(input_neg.data, quant=quant, prec=prec)
        self.weight.data = quantize(self.weight_org, quant=quant, prec=prec)
        out_pos, out_neg = linear_or_add(input_pos, input_neg, self.weight)
        return out_pos, out_neg
    
class Linear_Quant(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(Linear_Quant, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
    def forward(self, input, prec):
        input.data = quantize(input.data, quant=True, prec=prec)
        self.weight.data = quantize(self.weight_org, quant=True, prec=prec)
        out = linear_quant(input, self.weight, err=prec)
        return out
            
class Conv2d_Quant(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Quant, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
    def forward(self, input, prec):
        input.data = quantize(input.data, quant=True, prec=prec)
        self.weight.data = quantize(self.weight_org, quant=True, prec=prec)
        out = conv2d_quant(input, self.weight, padding=self.padding, stride=self.stride, err=prec)
        return out
        
class Norm_Shift(nn.Module):
    def __init__(self, momentum=0.1):
        super(Norm_Shift, self).__init__()
        self.mean = 1
        self.momentum = momentum
    def forward(self, input):
        shift = 2**np.ceil(np.log2(float(self.mean)))
        output = input / (2*shift)
        if self.training:
            with torch.no_grad():
                batch_mean = rms(input)
                self.mean = (1-self.momentum)*self.mean + self.momentum*batch_mean
        return output
    
def linear_xnor(activation, weight, bias, xnor=False, err=16):
    output = F.linear(activation, weight, bias)
    if xnor:
        err_mult = 2**err
        output.data = sc_extension.linear_xnor(activation, weight, err_mult)
        if bias is not None:
            output.data += bias.view(1,-1)
    return output
def linear_and(activation, weight, bias, sc=False, err=16):
    output = F.linear(activation, weight, bias)
    if sc:
        err_mult = 2**err
        output_data = sc_extension.linear_and(activation, weight, err_mult)
        output.data = output_data[0] - output_data[1]
        if bias is not None:
            output.data += bias.view(1,-1)
    return output

def conv2d_and(activation, weight, bias, padding=0, stride=1, sc=False, err=16):
    output = F.conv2d(activation, weight, bias, padding=padding, stride=stride)
    if sc:
        err_mult=2**err
        output_data = sc_extension.conv2d_and(activation, weight, err_mult, padding, stride)
        output.data = output_data[0] - output_data[1]
        if bias is not None:
            output.data += bias.view(1,-1,1,1)
    return output

def conv2d_xnor(activation, weight, bias, padding=0, stride=1, xnor=False, err=16):
    output = F.conv2d(activation, weight, bias, padding=padding, stride=stride)
    if xnor:
        err_mult = 2**err
        output.data = sc_extension.conv2d_xnor(activation, weight, err_mult, padding, stride)
        if bias is not None:
            output.data += bias.view(1,-1,1,1)
    return output

class Linear_And(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(Linear_And, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
    def forward(self, input, sc=False, err=16):
        self.weight.data = self.weight_org
        out = linear_and(input, self.weight, self.bias, sc=sc, err=err)
        return out
    
class Conv2d_And(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_And, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
    def forward(self, input, sc=False, err=16):
        self.weight.data = self.weight_org
        out = conv2d_and(input, self.weight, self.bias, padding=self.padding, stride=self.stride, sc=sc, err=err)
        return out

class Linear_Xnor(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(Linear_Xnor, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
    def forward(self, input, xnor=False, err=16):
        self.weight.data = self.weight_org
        out = linear_xnor(input, self.weight, self.bias, xnor=xnor, err=err)
        return out

class Conv2d_Xnor(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Xnor, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
    def forward(self, input, xnor=False, err=16):
        self.weight.data = self.weight_org
        out = conv2d_xnor(input, self.weight, self.bias, padding=self.padding, stride=self.stride, xnor=xnor, err=err)
        return out

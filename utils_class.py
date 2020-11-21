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
import utils_functional

import sc_extension

from sys import exit

sc_compute = '1d_bin'
global_gen = 'rand'
global_scaleonly = False
global_im2col = False
global_sync = True
global_dumpall = False
limit = 1

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}



def quantize(input, quant=False, prec=8):
    prec_2 = 2**prec
    if quant:
        input = (input * prec_2).round().clamp(-prec_2+1, prec_2-1)/prec_2
    return input

def quantize_r(input, quant=False, prec=8):
    prec_2 = 2**prec
    max_value = max(input.max(), -(input.min()))
#     print((input * prec_2 / max_value))
    if quant:
        input = (input * prec_2 / max_value).round().clamp(-prec_2, prec_2-1)*max_value/prec_2
    return input
            
class QuantizeConv2d(nn.Conv2d):
    '''
    Quantized conv2d with mask for pruning
    '''
    def __init__(self, *kargs, **kwargs):
        super(QuantizeConv2d, self).__init__(*kargs, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('weight_org', self.weight.data.clone())
    
    def forward(self, input, prec=7):
        # If mult exists, overwrites pruning
        input.data = quantize(input.data, quant=True, prec=prec)
        self.weight.data = quantize(self.weight_org, quant=True, prec=prec)
#         out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        out = utils_functional.conv2d_quant_trun(input, self.weight, self.padding, self.stride, prec=prec, trunc_add=False)
        return out
    
class QuantizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('weight_org', self.weight.data.clone())
    
    def forward(self, input, prec=7):
        # If mult exists, overwrites pruning
        input.data = quantize(input.data, quant=True, prec=prec)
        self.weight.data = quantize(self.weight_org, quant=True, prec=prec)
#         out = nn.functional.linear(input, self.weight)
        out = utils_functional.linear_quant_trun(input, self.weight, prec=prec, trunc_add=False)
        return out
    
class Conv2d_Or_Streamout_relu_pool(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        try:
            self.pool = kwargs['pool']
        except:
            self.pool = (2,2)
        else:
            del(kwargs['pool'])
            
        try:
            bn = kwargs['bn']
        except:
            bn = False
        else:
            del(kwargs['bn'])

        super(Conv2d_Or_Streamout_relu_pool, self).__init__(*kargs, **kwargs)
        if bn:
            self.bn = nn.BatchNorm2d(self.weight.size(0))
        else:
            self.bn = nn.Identity()
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    def forward(self, input, prec=7, err=7):
        self.weight.data = quantize(self.weight_org, quant=True, prec=prec)
        out = utils_functional.conv2d_generic(input, self.weight, bit_length=2**err, padding=self.padding, stride=self.stride, pool=self.pool, generator='lfsr', forward=sc_compute, relu=True, bn=self.bn, stream_out=True)
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
        out = utils_functional.conv2d_generic(input, self.weight, bit_length=2**err, padding=self.padding, stride=self.stride, generator='lfsr', forward=sc_compute, relu=False, stream_out=True, a_stream=False)
        return out
    
class Conv2d_Or_Stream_relu_pool(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        try:
            self.pool = kwargs['pool']
        except:
            self.pool = (2,2)
        else:
            del(kwargs['pool'])
            
        try:
            bn = kwargs['bn']
        except:
            bn = False
        else:
            del(kwargs['bn'])
            
        super(Conv2d_Or_Stream_relu_pool, self).__init__(*kargs, **kwargs)
        if bn:
            self.bn = nn.BatchNorm2d(self.weight.size(0))
        else:
            self.bn = nn.Identity()
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    def forward(self, input):
        self.weight.data = quantize(self.weight_org, quant=True, prec=7)
        out = utils_functional.conv2d_generic(input, self.weight, bit_length=2**7, padding=self.padding, stride=self.stride, pool=self.pool, generator='lfsr', forward=sc_compute, relu=True, bn=self.bn, stream_out=True, a_stream=True)
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
        out = utils_functional.conv2d_generic(input, self.weight, padding=self.padding, stride=self.stride, generator='lfsr', forward=sc_compute, relu=False, stream_out=True, a_stream=True)
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
        input.data = quantize(input.data, quant=True, prec=prec)
        self.weight.data = quantize(self.weight_org, quant=True, prec=prec)
#         self.weight.data = self.weight_org.sign()
        out = utils_functional.linear_generic(input, self.weight, bit_length=2**err, forward='full_bin', sync=False, generator=global_gen)
#         out = linear_add(input, self.weight, bit_length=2**err)
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
        input.data = quantize(input.data, quant=True, prec=prec)
        self.weight.data = quantize(self.weight_org, quant=True, prec=prec)
#         self.weight.data = self.weight_org.sign()
        out = utils_functional.conv2d_generic(input, self.weight, bit_length=2**err, padding=self.padding, stride=self.stride, forward='full_bin', sync=False, generator=global_gen)
#         out = conv2d_add(input, self.weight, bit_length=2**err, padding=self.padding, stride=self.stride)
        return out

class Conv2d_Add_Partial(nn.Conv2d):
    '''
    SC Conv2d using partial binary add
    '''
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Add_Partial, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    
    def forward(self, input, prec=7, err=7, bn=nn.Identity(), forward='1d_bin', generator='lfsr', z_unit=8, legacy=False, load_unit=8, load_wait_w=2, load_wait_a=2):
        input.data = quantize(input.data, quant=True, prec=prec)
        self.weight.data = quantize(self.weight_org, quant=True, prec=prec)
        if self.training:
            sync=global_sync
        else:
            sync=True
        out = utils_functional.conv2d_generic(input, self.weight, bit_length=2**err, padding=self.padding, stride=self.stride, forward=forward, sync=sync, generator=generator, im2col=global_im2col, bn=bn, legacy=legacy, z_unit=z_unit, load_unit=load_unit, load_wait_w=load_wait_w, load_wait_a=load_wait_a)
        if global_dumpall:
            conv_count=1
            while os.path.exists("./dump_dump/conv_{0}_in.npy".format(conv_count)):
                conv_count += 1
            np.save("./dump_dump/conv_{0}_in.npy".format(conv_count), input.data.numpy())
            np.save("./dump_dump/conv_{0}_weight.npy".format(conv_count), self.weight.data.numpy())
            np.save("./dump_dump/conv_{0}_out.npy".format(conv_count), out.data.numpy())
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
        out = utils_functional.conv2d_generic(input, self.weight, padding=self.padding, stride=self.stride, bit_length=2**err, forward='full_or', generator=global_gen)
        return out

class Linear_Or_Stream(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(Linear_Or_Stream, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    def forward(self, input):
        self.weight.data = quantize(self.weight_org, quant=True, prec=7)
        out = utils_functional.linear_generic(input, self.weight, a_stream=True, stream_out=True, forward='full_or')
        return out

class Linear_Or(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(Linear_Or, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    
    def forward(self, input, add_or=False, prec=7, err=7, add_full=1, true_or=False, add_count=False, generator='lfsr'):
        if prec is not None:
            quant=True
        else:
            quant=False
            prec=8
        add_or = self.add_or or add_or
        input.data = quantize(input.data, quant=quant, prec=prec)
        self.weight.data =quantize(self.weight_org, quant=quant, prec=prec)
        out = utils_functional.linear_generic(input, self.weight, bit_length=2**err, forward='full_or', generator=generator)
        if global_dumpall:
            conv_count=1
            while os.path.exists("./dump_dump/fc_{0}_in.npy".format(conv_count)):
                conv_count += 1
            np.save("./dump_dump/fc_{0}_in.npy".format(conv_count), input.data.numpy())
            np.save("./dump_dump/fc_{0}_weight.npy".format(conv_count), self.weight.data.numpy())
            np.save("./dump_dump/fc_{0}_out.npy".format(conv_count), out.data.numpy())
        return out
    
class Linear_Or_Unsync(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(Linear_Or_Unsync, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.add_or = False
    def forward(self, input, add_or=False, prec=7, err=7):
        quant=True
#         input.data = quantize(input.data, quant=quant, prec=prec)
#         self.weight.data = quantize(self.weight_org, quant=quant, prec=prec)
        self.weight.data = F.hardtanh(self.weight_org, -1,1)
        if self.training:
            sync = False
        else:
            sync = True
        out = utils_functional.linear_generic(input, self.weight, bit_length=2**err, forward='full_or', generator=global_gen, sync=sync)
        return out
    
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

class BatchNorm2d_fixed(nn.BatchNorm2d):
    def __init__(self, *kargs, **kwargs):
        super(BatchNorm2d_fixed, self).__init__(*kargs, **kwargs)
        self.register_buffer('scale', None)
    def forward(self, x):
#         self.eval()
        if global_scaleonly:
            if self.training:
                scale_cur = 0.5/torch.sqrt(torch.mean(x.data**2))
                self.scale = self.scale*0.9 + scale_cur*0.1
            scale_quant = 2**torch.ceil(torch.log2(self.scale))
            out = x * scale_quant.data
            return out
#             return x
        else:
            out = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=self.training)
            if self.training:
                mean = x.mean(dim=(0,2,3))
                var = x.var(dim=(0,2,3), unbiased=False)
            else:
                mean = self.running_mean
                var = self.running_var
            if self.affine:
                weight = self.weight.data
                bias = self.bias.data
            else:
                weight = 1
                bias = 0
            w_n = weight/torch.sqrt(var + self.eps)
            b_n = bias - mean*weight/torch.sqrt(var + self.eps)
            w_n, self.scale = utils_functional.quantize_shift(w_n.detach())
            b_n, _ = utils_functional.quantize_shift(b_n.detach(), self.scale)
            w_n = w_n.reshape(w_n.size(0),1,1)
            b_n = b_n.reshape(b_n.size(0),1,1)
    #         print("Before")
    #         print(out.data[0,0,:4,:4])

            
            out.data = (x.data*w_n + b_n).to(out.dtype)
            if global_dumpall:
                bn_count=1
                while os.path.exists("./dump_dump/bn2d_{0}_w.npy".format(bn_count)):
                    bn_count += 1
                np.save("./dump_dump/bn2d_{0}_w.npy".format(bn_count), w_n.numpy())
                np.save("./dump_dump/bn2d_{0}_b.npy".format(bn_count), b_n.numpy())
                np.save("./dump_dump/bn2d_{0}_in.npy".format(bn_count), x.data.numpy())
                np.save("./dump_dump/bn2d_{0}_out.npy".format(bn_count), out.data.numpy())
    #         print("After")
    #         print(out.data[0,0,:4,:4], '\n')
            return out
    
class BatchNorm1d_fixed(nn.BatchNorm1d):
    def __init__(self, *kargs, **kwargs):
        super(BatchNorm1d_fixed, self).__init__(*kargs, **kwargs)
        self.register_buffer('scale', None)
    def forward(self, x):
#         self.eval()
        out = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=self.training)
        if self.training:
            mean = x.mean(dim=(0))
            var = x.var(dim=(0), unbiased=False)
        else:
            mean = self.running_mean
            var = self.running_var
        if self.affine:
            weight = self.weight.data
            bias = self.bias.data
        else:
            weight = 1
            bias = 0
        w_n = weight/torch.sqrt(var + self.eps)
        b_n = bias - mean*weight/torch.sqrt(var + self.eps)
        w_n, self.scale = utils_functional.quantize_shift(w_n)
        b_n, _ = utils_functional.quantize_shift(b_n, self.scale)
        
        out.data = (x.data*w_n + b_n).to(out.dtype)
        if global_dumpall:
            bn_count=1
            while os.path.exists("./dump_dump/bn1d_{0}_w.npy".format(bn_count)):
                bn_count += 1
            np.save("./dump_dump/bn1d_{0}_w.npy".format(bn_count), w_n.numpy())
            np.save("./dump_dump/bn1d_{0}_b.npy".format(bn_count), b_n.numpy())
            np.save("./dump_dump/bn1d_{0}_in.npy".format(bn_count), x.data.numpy())
            np.save("./dump_dump/bn1d_{0}_out.npy".format(bn_count), out.data.numpy())
#         print(self.scale.item())
        return out

class Binary_Select(nn.Module):
    def __init__(self):
        super(Binary_Select, self).__init__()
        self.choose_0 = nn.Parameter(torch.tensor([0.]))
#         self.register_parameter('choose_0', torch.tensor([1]))
    def forward(self, x0, x1):
        choose_exp = torch.exp(self.choose_0)
        choose_0 = choose_exp / (1+choose_exp)
        x = x0*choose_0 + x1*(1-choose_0)
#         if self.training:
#             # randomized choice
#             rand = torch.rand(1, device=x0.device)
#             if choose_0>rand:
#                 x.data = x0.data
#             else:
#                 x.data = x1.data
#         else:
#             if choose_0>0.5:
#                 x.data = x0.data
#             else:
#                 x.data = x1.data
                
        if choose_0>0.5:
            x.data = x0.data
        else:
            x.data = x1.data
        return x
    
# class Conv2d_Or_Stream(nn.Conv2d):
#     '''
#     Or-based conv2d without conversion and using stream input
#     '''
#     def __init__(self, *kargs, **kwargs):
#         super(Conv2d_Or_Stream, self).__init__(*kargs, **kwargs)
#         self.register_buffer('weight_org', self.weight.data.clone())
#         self.add_or = False
#     def forward(self, input):
#         self.weight.data = quantize(self.weight_org, quant=True, prec=7)
#         out = utils_functional.conv2d_generic(input, self.weight, padding=self.padding, stride=self.stride, generator='lfsr', forward=sc_compute, relu=False, stream_out=True, a_stream=True)
#         return out
    
class Conv2d_Approx(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        try:
            err = kwargs['err']
        except:
            err = torch.zeros(128,128)
        else:
            del(kwargs['err'])
        super(Conv2d_Approx, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.register_buffer('err', err)
    def forward(self, input, err_prof):
        self.weight.data = quantize(self.weight_org, quant=True, prec=7)
        input.data = quantize(input.data, quant=True, prec=7)
        out = F.conv2d(input, self.weight, padding=self.padding, stride=self.stride)
#         if self.training:
#             err_avg = (err_prof.mean() * self.weight.data.sign().sum((1,2,3), keepdim=True).squeeze(-1)) / (128*128*2)
#             out = out + err_avg
#         else:
#         out.data = utils_functional.conv2d_approx(input.data, self.weight.data, padding=self.padding, stride=self.stride, err_prof=err_prof, train=self.training)
        return out
    
class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):
#         if (input.size(1) != 3) and (input.size(1) != 1):
#             input.data = Binarize(input.data)
        input.data = quantize(input.data, quant=True, prec=7)
        self.weight.data=utils_functional.Binarize(self.weight_org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

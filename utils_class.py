import torch
import torch.nn as nn
import torch.nn.functional as F


import utils_functional

def quantize(input, prec=8):
    prec_2 = 2**prec
    input = (input * prec_2).round().clamp(-prec_2+1, prec_2-1)/prec_2
    return input

class Conv2d_Add_Partial(nn.Conv2d):
    '''
    SC Conv2d using partial binary add
    '''
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Add_Partial, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
    
    def forward(self, input, prec=7, err=7, bn=nn.Identity(), forward='1d_bin', generator='lfsr', z_unit=8, legacy=False, load_unit=8, load_wait_w=2, load_wait_a=2):
        input.data = quantize(input.data, prec=prec)
        self.weight.data = quantize(self.weight_org, prec=prec)
        out = utils_functional.conv2d_generic(input, self.weight, bit_length=2**err, padding=self.padding, stride=self.stride, forward=forward, generator=generator, bn=bn, legacy=legacy, z_unit=z_unit, load_unit=load_unit, load_wait_w=load_wait_w, load_wait_a=load_wait_a)
        return out

class BatchNorm2d_fixed(nn.BatchNorm2d):
    def __init__(self, *kargs, **kwargs):
        super(BatchNorm2d_fixed, self).__init__(*kargs, **kwargs)
        self.register_buffer('scale', None)
    def forward(self, x):
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
        out.data = (x.data*w_n + b_n).to(out.dtype)
        return out
    
class BatchNorm1d_fixed(nn.BatchNorm1d):
    def __init__(self, *kargs, **kwargs):
        super(BatchNorm1d_fixed, self).__init__(*kargs, **kwargs)
        self.register_buffer('scale', None)
    def forward(self, x):
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
        return out
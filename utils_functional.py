import torch

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

'''
Accelerated kernels
'''
import sc_extension
import sc_extension_cuda

'''
File containing SC-specific function implementations
'''
# Using torch.float16 for compute_type and compare_type improves performance when using legacy computation on 
# GPUs supporting half precision, but may cause issue with CPU implementation and older GPUs
compute_type = torch.float32
compare_type = torch.float32
# share_more and share_max increases sharing to higher levels than is optimal. Change to True to observe the 
# effect of too much sharing
global_share_more = False
global_share_max = False
# Use true or for training instead of approximation using activation function. Has high performance penalties
global_trueor = False
# Default precision is 8-bit integer, but one bit is used for sign, so 7 bits are left
prec_default = 7

'''
Helper functions
'''
def quantize(input, prec=8):
    '''
    Quantize values between 0 and 1
    '''
    prec_2 = 2**prec
    input = (input * prec_2).round().clamp(-prec_2+1, prec_2-1)/prec_2
    return input

def quantize_shift(tensor, scale=None, scale_only=False):
    '''
    Quantize values with a shift to adjust range
    '''
    if scale_only:
        scale = torch.mean(tensor)*2
        scale = 2**torch.ceil(torch.log2(scale))
        tensor_quant = torch.ones_like(tensor)*scale
        return tensor_quant, scale
    else:
        if scale is None:
            scale = torch.mean(tensor)*3
            scale = 2**torch.ceil(torch.log2(scale))
        tensor_quant = tensor / scale
        tensor_quant = (tensor_quant * 128).round().clamp(-127, 127)/128
        tensor_quant = tensor_quant * scale
        return tensor_quant, scale  

'''
Generator functions
'''

def acc_init(w_pos, w_neg, a=None, device=torch.device('cpu'), prec=128):
    '''
    Initialize generator for accurate random generation
    '''
    w_pos_cur = w_pos.clone()
    w_neg_cur = w_neg.clone()
    if a is not None:
        a_cur = a.clone()
        return a_cur, w_pos_cur, w_neg_cur
    else:
        return None, w_pos_cur, w_neg_cur

def lfsr_init(w_size, a_size=None, device=torch.device('cpu'), prec=128):
    '''
    Initialize generator for LFSR
    '''
    weight_split_size_flat = np.array(w_size).prod()
    weight_seed_pos = np.arange(67, weight_split_size_flat+67)%(prec-1)+1
    weight_seed_neg = np.arange(37, weight_split_size_flat+37)%(prec-1)+1
    rand_weight_pos = torch.from_numpy(weight_seed_pos).reshape(w_size).to(device)
    rand_weight_neg = torch.from_numpy(weight_seed_neg).reshape(w_size).to(device)
    if a_size is not None:
        input_split_size_flat = np.array(a_size).prod()
        input_seed = np.arange(0, input_split_size_flat)%(prec-1)+1
        rand_input = torch.from_numpy(input_seed).reshape(a_size).to(device)
        return rand_input, rand_weight_pos, rand_weight_neg
    else:
        return None, rand_weight_pos, rand_weight_neg

def rand_init(w_size, a_size=None, device=torch.device('cpu'), prec=128):
    '''
    Initialize generator for simulated true random generation
    '''
    rand_weight_pos = torch.randint(prec, w_size, dtype=compare_type, device=device)
    rand_weight_neg = torch.randint(prec, w_size, dtype=compare_type, device=device)
    if a_size is not None:
        rand_input = torch.randint(prec, a_size, dtype=compare_type, device=device)
        return rand_input, rand_weight_pos, rand_weight_neg
    else:
        return None, rand_weight_pos, rand_weight_neg
    
def lfsr_5(rand_in):
    '''
    5-bit LFSR
    '''
    rand_out = ((rand_in//16)+(rand_in//4)%2)%2+2*(rand_in%16)
    return rand_out

def lfsr_6(rand_in):
    '''
    6-bit LFSR
    '''
    rand_out = ((rand_in//32)+(rand_in//16)%2)%2+2*(rand_in%32)
    return rand_out
    
def lfsr_7(rand_in):
    '''
    7-bit LFSR
    '''
    rand_out = ((rand_in//32)%2+rand_in//64)%2+2*(rand_in%64)
    return rand_out

def lfsr_7_xnor(rand_in):
    '''
    7-bit LFSR using xnor instead of xor
    '''
    rand_out = 1-((rand_in//32)%2+rand_in//64)%2+2*(rand_in%64)
    return rand_out

def lfsr_8(rand_in):
    '''
    8-bit LFSR
    '''
    rand_out = ((rand_in//128)+(rand_in//32)%2+(rand_in//16)%2+(rand_in//8)%2)%2+2*(rand_in%128)
    return rand_out
    
def lfsr_cont(rand_input, rand_weight_pos, rand_weight_neg, bit_length=128):
    '''
    Continue generation using LFSR
    '''
    if bit_length==128:
        lfsr_gen = lfsr_7
    elif bit_length==32:
        lfsr_gen = lfsr_5
    elif bit_length==64:
        lfsr_gen = lfsr_6
    elif bit_length==256:
        lfsr_gen = lfsr_8
    if rand_input is not None:
        rand_input = lfsr_gen(rand_input).to(compare_type)
    rand_weight_pos = lfsr_gen(rand_weight_pos).to(compare_type)
    rand_weight_neg = lfsr_gen(rand_weight_neg).to(compare_type)
    return rand_input, rand_weight_pos, rand_weight_neg

def acc_cont(input_cur, weight_pos_cur, weight_neg_cur, device, k, prec=128):
    '''
    Continue generation using accurate random generator
    '''
    weight_pos_norm = weight_pos_cur * prec / (prec-k)
    weight_neg_norm = weight_neg_cur * prec / (prec-k)
    rand_pos = torch.randint(prec, weight_pos_cur.size(), dtype=compare_type, device=device)
    rand_neg = torch.randint(prec, weight_neg_cur.size(), dtype=compare_type, device=device)
    w_pos_bit = (weight_pos_norm > rand_pos).to(compute_type)
    w_neg_bit = (weight_neg_norm > rand_neg).to(compute_type)
    weight_pos_cur -= w_pos_bit
    weight_neg_cur -= w_neg_bit
    if input_cur is not None:
        input_norm = input_cur * prec / (prec-k)
        rand_input = torch.randint(prec, input_cur.size(), dtype=compare_type, device=device)
        a_bit = (input_norm > rand_input).to(compute_type)
        input_cur -= a_bit
        return a_bit, w_pos_bit, w_neg_bit, input_cur, weight_pos_cur, weight_neg_cur
    else:
        return None, w_pos_bit, w_neg_bit, None, weight_pos_cur, weight_neg_cur
'''
Forward functions for training
'''
def linear_or_approx(activation, weight, true_or=(False or global_trueor)):
    '''
    Floating-point forward function to guide back propagation for linear layers
    '''
    if true_or:
        mult_result = activation.unsqueeze(1)*weight
        return 1-torch.prod(1-mult_result, dim=-1)
    else:
        return 1-torch.exp(-F.linear(activation, weight))
def conv2d_or_approx(activation, weight, padding, stride, true_or=(False or global_trueor)):
    '''
    Floating-point forward function to guide back propagation for conv2d layers using full or accumulation
    '''
    if true_or:
        # True or is achieved by first performing an im2col transformation
        kernel_size = weight.size()[-2:]
        activation_col = F.unfold(activation, kernel_size, dilation=1, padding=padding, stride=stride)
        weight_col = weight.view(weight.data.size(0),-1)
        a_size = list(activation_col.size())
        a_size.insert(1,1)
        w_size = list(weight_col.size())
        w_size.append(1)
        activation_col = activation_col.view(a_size)
        weight_col = weight_col.view(w_size)
    
        mult_result = activation_col*weight_col
        add_res = 1-torch.prod(1-mult_result, dim=2)
        size_out = np.sqrt(add_res.size(-1)).astype(int)
        return F.fold(add_res, (size_out, size_out), (1,1))
    else:
        return 1-torch.exp(-F.conv2d(activation, weight, padding=padding, stride=stride))
        
def conv2d_or_bin_2d(activation, w_pos, w_neg, padding, stride):
    '''
    Floating-point forward function to guide back propagation for conv2d layers using mixed accumulation
    Accumulation is done using fixed-point adders for x and y dimensions, and simulated using multiple convolutions
    '''
    i_x = activation.size(2)
    i_y = activation.size(3)
    f_x = w_pos.size(2)
    f_y = w_pos.size(3)
    result_pos_value = []
    result_neg_value = []
    for i in range(f_x):
        for j in range(f_y):
            a_sec = activation[...,i:i_x-f_x+i+1,j:i_y-f_y+j+1].clone()
            w_pos_sec = w_pos[...,i:i+1,j:j+1].clone()
            w_neg_sec = w_neg[...,i:i+1,j:j+1].clone()
            result_pos_value.append(1-torch.exp(-F.conv2d(a_sec, w_pos_sec, stride=stride)))
            result_neg_value.append(1-torch.exp(-F.conv2d(a_sec, w_neg_sec, stride=stride)))
    
    result_pos_value = torch.stack(result_pos_value, 0).sum(0)
    result_neg_value = torch.stack(result_neg_value, 0).sum(0)
    return result_pos_value, result_neg_value

def conv2d_or_bin_1d(activation, w_pos, w_neg, padding, stride):
    '''
    Floating-point forward function to guide back propagation for conv2d layers using mixed accumulation
    Accumulation is done using fixed-point adders for the y dimension, and simulated using multiple convolutions
    '''
    i_y = activation.size(3)
    f_y = w_pos.size(3)
    result_pos_value = []
    result_neg_value = []
    for j in range(f_y):
        a_sec = activation[...,j:i_y-f_y+j+1].clone()
        w_pos_sec = w_pos[...,j:j+1].clone()
        w_neg_sec = w_neg[...,j:j+1].clone()
        result_pos_value.append(1-torch.exp(-F.conv2d(a_sec, w_pos_sec, stride=stride)))
        result_neg_value.append(1-torch.exp(-F.conv2d(a_sec, w_neg_sec, stride=stride)))
    result_pos_value = torch.stack(result_pos_value, 0).sum(0)
    result_neg_value = torch.stack(result_neg_value, 0).sum(0)
    return result_pos_value, result_neg_value

def conv2d_or_bin_z(activation, w_pos, w_neg, padding, stride):
    '''
    Floating-point forward function to guide back propagation for conv2d layers using mixed accumulation
    Accumulation is done using fixed-point adders for the z dimension, and simulated using multiple convolutions
    '''
    c_in = activation.size(1)
    result_pos_value = []
    result_neg_value = []
    for c in range(c_in):
        a_sec = activation[:,c:c+1].clone()
        w_pos_sec = w_pos[:,c:c+1].clone()
        w_neg_sec = w_neg[:,c:c+1].clone()
        result_pos_value.append(1-torch.exp(-F.conv2d(a_sec, w_pos_sec, stride=stride)))
        result_neg_value.append(1-torch.exp(-F.conv2d(a_sec, w_neg_sec, stride=stride)))
    result_pos_value = torch.stack(result_pos_value, 0).sum(0)
    result_neg_value = torch.stack(result_neg_value, 0).sum(0)
    return result_pos_value, result_neg_value

def conv2d_or_bin_yz(activation, w_pos, w_neg, padding, stride, z_unit):
    '''
    Floating-point forward function to guide back propagation for conv2d layers using mixed accumulation
    Accumulation is done using fixed-point adders for y and z dimension, and simulated using multiple convolutions
    y dimension is accumulated using fixed-point adders only, while z dimension performs some accumulations using OR to reduce cost, specified by z_unit argument
    '''
    c_in = activation.size(1)
    i_y = activation.size(3)
    f_y = w_pos.size(3)
    result_pos_value = []
    result_neg_value = []# SC
    for c in range(0,c_in,z_unit):
        if c+z_unit<=c_in:
            c_end=c+z_unit
        else:
            c_end=c_in
        for j in range(f_y):
            a_sec = activation[:,c:c_end,:,j:i_y-f_y+j+1].clone()
            w_pos_sec = w_pos[:,c:c_end,:,j:j+1].clone()
            w_neg_sec = w_neg[:,c:c_end,:,j:j+1].clone()
            result_pos_value.append(1-torch.exp(-F.conv2d(a_sec, w_pos_sec, stride=stride)))
            result_neg_value.append(1-torch.exp(-F.conv2d(a_sec, w_neg_sec, stride=stride)))
    result_pos_value = torch.stack(result_pos_value, 0).sum(0)
    result_neg_value = torch.stack(result_neg_value, 0).sum(0)
    return result_pos_value, result_neg_value
    
'''
Generic functional layers. All other configurations should be derived from this
'''

def linear_generic(activation, weight, **kwargs):
    '''
    Generic linear layer
    Arguments:
    bit_length: stream length to use
    prec: weight and activation quantization precision specified using number of allowed discrete values
    share: allow limited sharing of stream generators to reduce cost and improve accuracy
    generator: stream generator to use
    forward: sc computation to use
    '''
    try:
        bit_length = kwargs['bit_length']
    except:
        bit_length = 128
    
    try:
        prec = kwargs['prec']
    except:
        prec = 128
        
    try:
        share = kwargs['share']
    except:
        share = True
        
    try:
        generator = kwargs['generator']
    except:
        generator = None
        
    try:
        forward = kwargs['forward']
    except:
        forward = 'full_or'
    
    device = activation.device
    bit_range = prec-1
    
    # Quantization precision is tied to stream length for LFSR generator. E.g.: 5-bit precision is used for 
    # 32-bit streams (+1 bit precision for sign)
    if generator=='lfsr':
        prec = bit_length
    input_split = (activation.data*prec).to(compare_type)
    w_pos_split = (weight.data*prec).clamp(0,bit_range).to(compare_type)
    w_neg_split = -(weight.data*prec).clamp(-bit_range,0).to(compare_type)

    # Share stream generator between different filters and different inputs if permitted
    if share:
        a_size = [activation.size(-1)]
        w_size = [weight.size(-1)]
    else:
        a_size = activation.size()
        w_size = weight.size()

    # Only LFSR and true random generator is implemented for FC layers (for now)
    if generator=='lfsr':
        rand_input, rand_weight_pos, rand_weight_neg = lfsr_init(w_size, a_size, device, prec)
    else:
        rand_input, rand_weight_pos, rand_weight_neg = rand_init(w_size, a_size, device, prec)

    result_pos = []
    result_neg = []

    for k in range(bit_length):
        # SC computation is simulated as sum of normal FC layers on single bits
        if generator=='lfsr':
            rand_input, rand_weight_pos, rand_weight_neg = lfsr_cont(rand_input, rand_weight_pos, rand_weight_neg, bit_length=bit_length)
        else:
            rand_input, rand_weight_pos, rand_weight_neg = rand_init(w_size, a_size, device, prec)

        a_bit = (input_split > rand_input).to(compute_type)
        w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
        w_neg_bit = (w_neg_split > rand_weight_neg).to(compute_type)
        # For OR accumulation, having one 1 in the entire accumulation means output is one, so taking the sign of 
        # normal accumulation is equivalent to doing OR accumulation
        if forward == 'full_or':
            result_pos.append(F.linear(a_bit, w_pos_bit).sign())
            result_neg.append(F.linear(a_bit, w_neg_bit).sign())
        elif forward == 'full_bin':
            result_pos.append(F.linear(a_bit, w_pos_bit))
            result_neg.append(F.linear(a_bit, w_neg_bit))
    result_pos = torch.stack(result_pos, 0)
    result_neg = torch.stack(result_neg, 0)
    
    w_pos = weight.clamp(0,100)
    w_neg = -(weight.clamp(-100,0))
    activation = activation.to(w_pos.dtype)
    
    # Floating point forward pass to guide backpropagation
    if forward == 'full_or':
        result_pos_value = linear_or_approx(activation, w_pos)
        result_neg_value = linear_or_approx(activation, w_neg)
    elif forward == 'full_bin':
        result_pos_value = F.linear(activation, w_pos)
        result_neg_value = F.linear(activation, w_neg)
        
    device = str(result_pos_value.device)[-1]
        
    # Result from SC computation overwrites floating point forward pass
    result_pos_value.data = result_pos.mean(0)
    result_neg_value.data = result_neg.mean(0)
        
    return result_pos_value - result_neg_value

def conv2d_generic(activation, weight, padding, stride, **kwargs):
    '''
    Generic conv2d layer
    Arguments:
    bit_length: stream length to use
    prec: weight and activation quantization precision specified using number of allowed discrete values
    share: allow limited sharing of stream generators to reduce cost and improve accuracy
    generator: stream generator to use
    forward: sc computation to use
    legacy: use older implementation without optimization
    load_unit: number of bits to load each time for progressive loading
    load_wait_w: number of cycles to wait between loading weights for progressive loading
    load_wait_a: number of cycles to wait between loading activations for progressive loading
    z_unit: number of input channels to accumulate using OR
    '''
    try:
        bit_length = kwargs['bit_length']
    except:
        bit_length = 128
    
    try:
        prec = kwargs['prec']
    except:
        prec = 128
        
    try:
        acc = kwargs['acc']
    except:
        acc = False
        
    try:
        share = kwargs['share']
    except:
        share = True
        
    try:
        generator = kwargs['generator']
    except:
        generator = None
        
    try:
        forward = kwargs['forward']
    except:
        forward = 'full_or'
     
    try:
        legacy = kwargs['legacy']
    except:
        legacy = False
        
    try:
        load_unit = kwargs['load_unit']
    except:
        load_unit = 8
    try:
        load_wait_w = kwargs['load_wait_w']
    except:
        load_wait_w = 1
    try:
        load_wait_a = kwargs['load_wait_a']
    except:
        load_wait_a = 1
    '''
    Conv2d specific
    '''
    try:
        z_unit = kwargs['z_unit']
    except:
        z_unit = 1024
    
    '''
    End of Conv2d specific
    '''
    
    device = activation.device
    bit_range = prec-1
    cout = weight.size(0)
    
    # Quantization precision is tied to stream length for LFSR generator. E.g.: 5-bit precision is used for 
    # 32-bit streams (+1 bit precision for sign)
    if generator=='lfsr':
        prec = bit_length

    if (forward=='1d_bin') and (not legacy):
        if generator=='lfsr':
            if not activation.is_cuda:
                activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
                result_pos = sc_extension.conv2d_add_partial_new(activation.data, weight.data, bit_length, (0,0), stride).float()
                result_neg = torch.zeros_like(result_pos)
            else:
                activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
                result_pos = sc_extension_cuda.conv2d_add_partial_variable_acc(activation.data, weight.data, bit_length, (0,0), stride, prec_default, load_unit, load_wait_w, False, 2).float()
                result_neg = torch.zeros_like(result_pos)
        elif generator=='acc':
            activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
            result_pos = sc_extension_cuda.conv2d_add_partial_acc_acc(activation.data, weight.data, bit_length, (0,0), stride, share, 2).float()
            result_neg = torch.zeros_like(result_pos)
    elif (forward=='z_bin') and (activation.is_cuda) and (not legacy):
        activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
        result_pos = sc_extension_cuda.conv2d_add_partial_variable_acc(activation.data, weight.data, bit_length, (0,0), stride, prec_default, load_unit, load_wait_w, False, 0).float()
        result_neg = torch.zeros_like(result_pos)
    elif (forward=='yz_bin') and (activation.is_cuda) and (not legacy):
        activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
        result_pos = sc_extension_cuda.conv2d_add_yz_variable_acc(activation.data, weight.data, bit_length, (0,0), stride, prec_default, load_unit, load_wait_w, load_wait_a, False, z_unit).float()
        result_neg = torch.zeros_like(result_pos)
    elif (forward=='full_or') and (activation.is_cuda) and (not legacy):
        activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
        result_pos = sc_extension_cuda.conv2d_or_acc(activation.data, weight.data, bit_length, (0,0), stride, False).float()
        result_neg = torch.zeros_like(result_pos)
    else:
        activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))

        result_pos = 0
        result_neg = 0

        # Quantize activations and weights
        input_split = (activation.data*prec).to(compare_type)
        w_pos_split = (weight.data*prec).clamp(0,bit_range).to(compare_type)
        w_neg_split = -(weight.data*prec).clamp(-bit_range,0).to(compare_type)

        # Hybrid SC-fixed accumulation is achieved by splitting one conv into smaller convs
        # Each small one is done using OR accumulation, and they are added together in the end
        if forward=='2d_bin':
            i_x = activation.size(2)
            i_y = activation.size(3)
            f_x = weight.size(2)
            f_y = weight.size(3)
            f_size = f_x*f_y
            w_pos_split_temp = []
            w_neg_split_temp = []
            for l in range(f_x):
                for m in range(f_y):
                    w_pos_split_temp.append(w_pos_split[...,l:l+1,m:m+1])
                    w_neg_split_temp.append(w_neg_split[...,l:l+1,m:m+1])
            w_pos_split = torch.cat(w_pos_split_temp, 0)
            w_neg_split = torch.cat(w_neg_split_temp, 0)
        elif forward=='1d_bin':
            i_y = activation.size(3)
            f_y = weight.size(3)
            w_pos_split_temp = []
            w_neg_split_temp = []
            for m in range(f_y):
                w_pos_split_temp.append(w_pos_split[...,m:m+1])
                w_neg_split_temp.append(w_neg_split[...,m:m+1])
            w_pos_split = torch.cat(w_pos_split_temp, 0)
            w_neg_split = torch.cat(w_neg_split_temp, 0)
            
        # Share stream generator between different filters and different inputs if permitted
        if share:
            if global_share_more:
                a_size = list(activation.size()[-1:])
                w_size = list(w_pos_split.size()[-1:])
            elif global_share_max:
                a_size = [1]
                w_size = [1]
            else:
                a_size = list(activation.size()[-3:])
                w_size = list(w_pos_split.size()[-3:])
        else:
            a_size = activation.size()
            w_size = w_pos_split.size()

        # Initialize stream generators
        if generator=='lfsr':
            rand_input, rand_weight_pos, rand_weight_neg = lfsr_init(w_size, a_size, device, prec)
        elif generator=='acc':
            input_cur, weight_pos_cur, weight_neg_cur = acc_init(w_pos_split, w_neg_split, input_split)
        else:
            rand_input, rand_weight_pos, rand_weight_neg = rand_init(w_size, a_size, device, prec)

        for k in range(bit_length):
            # Generate bits
            if generator=='acc':
                a_bit, w_pos_bit, w_neg_bit, input_cur, weight_pos_cur, weight_neg_cur = acc_cont(input_cur, weight_pos_cur, weight_neg_cur, device, k, prec)
            else:
                if generator=='lfsr':
                    rand_input, rand_weight_pos, rand_weight_neg = lfsr_cont(rand_input, rand_weight_pos, rand_weight_neg, bit_length=bit_length)
                else:
                    rand_input, rand_weight_pos, rand_weight_neg = rand_init(w_size, a_size, device, prec)
                w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
                w_neg_bit = (w_neg_split > rand_weight_neg).to(compute_type)
                a_bit = (input_split > rand_input).to(compute_type)

            # Perform normal convolution
            result_pos_temp = F.conv2d(a_bit, w_pos_bit, stride=stride)
            result_neg_temp = F.conv2d(a_bit, w_neg_bit, stride=stride)
            # Simulate effect of different accumulation schemes
            if not forward=='full_bin':
                result_pos_temp = result_pos_temp.sign()
                result_neg_temp = result_neg_temp.sign()

            if forward == '1d_bin':
                for m in range(f_y):
                    result_pos = result_pos + result_pos_temp[:,m*cout:(m+1)*cout,...,m:i_y-f_y+m+1]
                    result_neg = result_neg + result_neg_temp[:,m*cout:(m+1)*cout,...,m:i_y-f_y+m+1]
            elif forward == '2d_bin':
                for l in range(f_x):
                    for m in range(f_y):
                        index = l*f_y + m
                        result_pos = result_pos + result_pos_temp[:,index*cout:(index+1)*cout, l:i_x-f_x+l+1, m:i_y-f_y+m+1] 
                        result_neg = result_neg + result_neg_temp[:,index*cout:(index+1)*cout, l:i_x-f_x+l+1, m:i_y-f_y+m+1]
            elif (forward=='full_or') or (forward=='full_bin'):
                result_pos = result_pos + result_pos_temp
                result_neg = result_neg + result_neg_temp
                                
    w_pos = weight.clamp(0,100)
    w_neg = -(weight.clamp(-100,0))
    activation = activation.to(w_pos.dtype)
    
    # Floating point forward pass
    if forward == 'full_or':
        result_pos_value = conv2d_or_approx(activation, w_pos, (0,0), stride)
        result_neg_value = conv2d_or_approx(activation, w_neg, (0,0), stride)
    elif forward == 'full_bin':
        result_pos_value = F.conv2d(activation, w_pos, padding=(0,0), stride=stride)
        result_neg_value = F.conv2d(activation, w_neg, padding=(0,0), stride=stride)
    elif forward == '2d_bin':
        result_pos_value, result_neg_value = conv2d_or_bin_2d(activation, w_pos, w_neg, (0,0), stride)
    elif (forward == '1d_bin') or (forward == 'bin_mult'):
        result_pos_value, result_neg_value = conv2d_or_bin_1d(activation, w_pos, w_neg, (0,0), stride)
    elif forward == 'z_bin':
        result_pos_value, result_neg_value = conv2d_or_bin_z(activation, w_pos, w_neg, (0,0), stride)
    elif forward == 'yz_bin':
        result_pos_value, result_neg_value = conv2d_or_bin_yz(activation, w_pos, w_neg, (0,0), stride, z_unit)
        
    # Result from SC computation overwrites floating point forward pass
    result_pos_value.data = result_pos / bit_length
    result_neg_value.data = result_neg / bit_length
    return result_pos_value - result_neg_value

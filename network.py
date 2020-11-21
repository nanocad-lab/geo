import os
import torch

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import utils_own
import utils_class
import utils_functional

global_cnt = 0
global_dump = False
global_usebn = True
global_userelu = False
dump_param = False
dump_dir = './dump_svhn/'


class FC_small(nn.Module):
    def __init__(self, add_or=True):
        super(FC_small, self).__init__()

        self.add_or = add_or
        self.register_buffer('fc1_result', None)
        self.register_buffer('fc2_result', None)
        
        self.fc1 = nn.Linear(784, 100, bias=False)
        self.fc2 = nn.Linear(100, 10, bias=False)
    def forward(self, x):
        x = x.view(-1, 28*28)
        if self.add_or:
            x = utils_own.linear_or_subtract(x, self.fc1.weight)
        else:
            x = F.relu(F.linear(x, self.fc1.weight))
        self.fc1_result = x.data.clone()
        if self.add_or:
            x = utils_own.linear_or_subtract(x, self.fc2.weight)
        else:
            x = F.linear(x, self.fc2.weight)
        self.fc2_result = x.data.clone()
        return x
    
class FC_small_unsync(nn.Module):
    def __init__(self, add_or=True):
        super(FC_small_unsync, self).__init__()
        
        self.add_or = add_or
        self.register_buffer('fc1_result', None)
        self.register_buffer('fc2_result', None)
        
        self.fc1 = utils_class.Linear_Or_Unsync(3072, 100, bias=False)
        self.fc2 = utils_class.Linear_Or_Unsync(100, 10, bias=False)
#         self.fc1 = utils_class.Linear_Or_Unsync(784, 10, bias=False)
        
    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = self.fc1(x)
        x = F.hardtanh(x,0,1)
        x = self.fc2(x)
        return x
    
class CONV_mid(nn.Module):
    def __init__(self, add_or=True, max_pool=False):
        super(CONV_mid, self).__init__()
        
        self.add_or = add_or
        self.err= 7
        self.dropconv = nn.Dropout2d(p=0.2)
        self.conv1 = utils_own.Conv2d_Or(3, 128, kernel_size=3, padding=1, bias=False)
        self.conv2 = utils_own.Conv2d_Or(128, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = utils_own.Conv2d_Or(128, 256, kernel_size=3, padding=1, bias=False)
        self.conv4 = utils_own.Conv2d_Or(256, 256, kernel_size=3, padding=1, bias=False)
        self.conv5 = utils_own.Conv2d_Or(256, 512, kernel_size=3, padding=1, bias=False)
        self.conv6 = utils_own.Conv2d_Or(512, 512, kernel_size=3, padding=1, bias=False)
        self.fc1 = utils_own.Linear_Or(4*4*512, 10, bias=False)
        self.pool = nn.AvgPool2d(2)
        
        self.true_or = False
        
        self.bn = nn.BatchNorm1d(10, affine=False)
    
    def forward(self, x):
        x = F.relu(self.conv1(x, add_or=self.add_or, err=self.err, true_or=self.true_or))
        x = F.relu(self.conv2(x, add_or=self.add_or, err=self.err, true_or=self.true_or))
        x = self.dropconv(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x, add_or=self.add_or, err=self.err, true_or=self.true_or))
        x = F.relu(self.conv4(x, add_or=self.add_or, err=self.err, true_or=self.true_or))
        x = self.dropconv(x)
        x = self.pool(x)
        x = F.relu(self.conv5(x, add_or=self.add_or, err=self.err, true_or=self.true_or))
        x = F.relu(self.conv6(x, add_or=self.add_or, err=self.err, true_or=self.true_or))
        x = self.dropconv(x)
        x = self.pool(x)
        x = x.view(-1, 4*4*512)
        x = self.fc1(x, add_or=self.add_or, err=self.err)
        x = self.bn(x)
        return x

class CONV_tiny_quant(nn.Module):
    def __init__(self, uniform=False):
        super(CONV_tiny_quant, self).__init__()

        self.conv1 = utils_class.QuantizeConv2d(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.QuantizeConv2d(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_class.QuantizeConv2d(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1 = utils_class.QuantizeLinear(4*4*64, 10, bias=False)

        self.tanh = nn.Hardtanh()
        self.pool = nn.AvgPool2d(2)
        self.bn4 = nn.BatchNorm1d(10, affine=True)

        self.err_used = [3,3,3,3]
        if uniform:
            self.conv1.weight_org.uniform_(-1,1)
            self.conv2.weight_org.uniform_(-1,1)
            self.conv3.weight_org.uniform_(-1,1)
            self.fc1.weight_org.uniform_(-1,1)
            self.conv1.weight.data.uniform_(-1,1)
            self.conv2.weight.data.uniform_(-1,1)
            self.conv3.weight.data.uniform_(-1,1)
            self.fc1.weight.data.uniform_(-1,1)

    def forward(self, x):
        x = F.relu(self.conv1(x, prec=self.err_used[0]))
        x = self.pool(x)
        x = self.tanh(x)
#         x = self.tanh(self.pool(x))

        x = F.relu(self.conv2(x, prec=self.err_used[1]))
        x = self.pool(x)
        x = self.tanh(x)
#         x = self.tanh(self.pool(x))

        x = F.relu(self.conv3(x, prec=self.err_used[2]))
        x = self.pool(x)
        x = self.tanh(x)
#         x = self.tanh(self.pool(x))

        x = x.view(-1, 4*4*64)
        x = self.fc1(x, prec=self.err_used[3])
        x = self.bn4(x)
        return x

class CONV_tiny_relu(nn.Module):
    def __init__(self, add_or=True, max_pool=False):
        super(CONV_tiny_relu, self).__init__()
   

        self.add_or = add_or
        self.register_buffer('conv1_result', torch.zeros(256,32,32,32))
        self.register_buffer('conv2_result', torch.zeros(256,32,16,16))
        self.register_buffer('conv3_result', torch.zeros(256,64,8,8))
        self.register_buffer('fc1_result', torch.zeros(256,10))
        
        self.conv1 = utils_own.Conv2d_Or(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_own.Conv2d_Or(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_own.Conv2d_Or(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1 = utils_own.Linear_Or(4*4*64, 10, bias=False)
        
        self.fcx1 = nn.Linear(32,10,bias=False)
        self.fcx2 = nn.Linear(32,10,bias=False)
        
        self.tanh = nn.Hardtanh()
        self.softmax = nn.LogSoftmax(dim=-1)
        
        if max_pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.AvgPool2d(2)
            
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(10, affine=False)
        
        self.err = [7,7,7,7]
        self.true_or = False

                             
    def forward(self, x):
        x = self.pool(self.conv1(x, add_or = self.add_or, err=self.err[0], prec=7, add_full=1, true_or=self.true_or))
        x = self.tanh(x)
        if x.size(0)>16:
            self.conv1_result = x.data.clone()
        x = F.relu(x)
        
        x = self.pool(self.conv2(x, add_or = self.add_or, err=self.err[1], prec=7, add_full=1, true_or=self.true_or))
        x = self.tanh(x)
        if x.size(0)>16:
            self.conv2_result = x.data.clone()
        x = F.relu(x)
        
        x = self.pool(self.conv3(x, add_or = self.add_or, err=self.err[2], prec=7, add_full=1, true_or=self.true_or))
        x = self.tanh(x)
        if x.size(0)>16:
            self.conv3_result = x.data.clone()
        x = F.relu(x)
        
        x = x.view(-1, 4*4*64)
        x = self.fc1(x, add_or = self.add_or, err=self.err[3], prec=7, add_full=1, true_or=self.true_or)
#         x = self.tanh(x)
        
        if x.size(0)>16:
            self.fc1_result = x.data.clone()
        x = self.bn4(x)
        return x#, x1, x2, x3   
    
class CONV_tiny_stream_pool(nn.Module):
    def __init__(self, add_or=True, max_pool=False):
        super(CONV_tiny_stream_pool, self).__init__()
        
        self.add_or = add_or
        self.register_buffer('conv1_result', torch.zeros(256,32,32,32))
        self.register_buffer('conv2_result', torch.zeros(256,32,16,16))
        self.register_buffer('conv3_result', torch.zeros(256,64,8,8))
        self.register_buffer('fc1_result', torch.zeros(256,10))
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv1 = utils_class.Conv2d_Or_Streamout_relu_pool(3, 32, kernel_size=5, padding=2, bias=False, pool=(2,2), bn=True)
        self.conv2 = utils_class.Conv2d_Or_Stream_relu_pool(32, 32, kernel_size=5, padding=2, bias=False, pool=(2,2), bn=True)
        self.conv3 = utils_class.Conv2d_Or_Stream_relu_pool(32, 64, kernel_size=5, padding=2, bias=False, pool=(2,2), bn=True)
        self.fc1 = utils_class.Linear_Or_Stream(4*4*64, 10, bias=False)
        self.bn4 = nn.BatchNorm1d(10, affine=False)
        
        self.err = [7,7,7,7]
        self.prec = [7,7,7,7]
        self.true_or = False
        
    def forward(self, x):
#         print(self.bn2.running_mean, self.bn2.running_var, self.bn2.weight, self.bn2.bias)
        x = self.conv1(x, err=self.err[0], prec=self.prec[0])
        x = self.conv2(x)
        x,x_v = self.conv3(x)
        x = x.view(x.size(0)*x.size(1), x.size(2), 4*4*64)
        x_v = x_v.view(-1, 4*4*64)
        x = self.fc1((x,x_v))
        x = x[0][1] - x[1][1]
        x = self.bn4(x)
        return x
    
class CONV_tiny_stream(nn.Module):
    def __init__(self, add_or=True, max_pool=False):
        super(CONV_tiny_stream, self).__init__()
        
        self.add_or = add_or
        self.register_buffer('conv1_result', torch.zeros(256,32,32,32))
        self.register_buffer('conv2_result', torch.zeros(256,32,16,16))
        self.register_buffer('conv3_result', torch.zeros(256,64,8,8))
        self.register_buffer('fc1_result', torch.zeros(256,10))
        
        self.conv1 = utils_class.Conv2d_Or_Streamout(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.Conv2d_Or_Stream(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_class.Conv2d_Or_Stream(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1 = utils_class.Linear_Or_Stream(4*4*64, 10, bias=False)
        
        self.fcx1 = nn.Linear(32,10,bias=False)
        self.fcx2 = nn.Linear(32,10,bias=False)
        
        self.tanh = nn.Hardtanh(-0.5, 0.5)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        if max_pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.AvgPool2d(2)
            
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(10, affine=False)
        
        self.err = [7,7,7,7]
        self.prec = [7,7,7,7]
        self.true_or = False

                             
    def forward(self, x):
        x_pos, x_neg = self.conv1(x, err=self.err[0], prec=self.prec[0])
        x = utils_functional.subtract_relu(x_pos, x_neg)
        x = utils_functional.avgpool2d_stream(x)

        x_pos, x_neg = self.conv2(x)
        x = utils_functional.subtract_relu(x_pos, x_neg)
        x = utils_functional.avgpool2d_stream(x)

        x_pos, x_neg = self.conv3(x)
        x = utils_functional.subtract_relu(x_pos, x_neg)
        x, x_v = utils_functional.avgpool2d_stream(x)

        x = x.view(x.size(0), x.size(1), 4*4*64)
        x_v = x_v.view(-1, 4*4*64)
        x = self.fc1((x,x_v))
        
        x = x[0][1] - x[1][1]
        x = self.bn4(x)
        return x#, x1, x2, x3

class CONV_tiny_pool(nn.Module):
    def __init__(self, add_or=True, max_pool=False):
        super(CONV_tiny_pool, self).__init__()
   

        self.add_or = add_or
        self.register_buffer('conv1_result', torch.zeros(256,32,32,32))
        self.register_buffer('conv2_result', torch.zeros(256,32,16,16))
        self.register_buffer('conv3_result', torch.zeros(256,64,8,8))
        self.register_buffer('fc1_result', torch.zeros(256,10))
        
        self.conv1 = utils_own.Conv2d_Or(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_own.Conv2d_Or(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_own.Conv2d_Or(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1 = utils_own.Linear_Or(4*4*64, 10, bias=False)
        
        self.fcx1 = nn.Linear(32,10,bias=False)
        self.fcx2 = nn.Linear(32,10,bias=False)
        
        self.tanh = nn.Hardtanh(-0.5, 0.5)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        if max_pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.AvgPool2d(2)
            
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(10, affine=False)
        
        self.err = [7,7,7,7]
        self.prec = [7,7,7,7]
        self.true_or = False

                             
    def forward(self, x):
        config = "Medium_skip"
        bit_off= 1
        
        print("0", torch.sqrt(torch.mean(x**2)))
        print("1.weight", torch.sqrt(torch.mean(self.conv1.weight**2)))
        
        if config == "Normal":
            x = self.conv1(x, add_or = self.add_or, err=self.err[0], prec=self.prec[0], add_full=1, true_or=self.true_or)
            x = self.pool(x)
        else:
            output = []
            for i in range(4):
                x1 = self.conv1(x, add_or = self.add_or, err=self.err[0]-bit_off, prec=self.prec[0], add_full=1, true_or=self.true_or)
                output.append(x1)
            x = utils_own.avgpool2d_convert(output, size=2, config=config)
        x = F.relu(x)
        
        print("1", torch.sqrt(torch.mean(x**2)))
        print("2.weight", torch.sqrt(torch.mean(self.conv2.weight**2)))

        if config == "Normal":
            x = self.conv2(x, add_or = self.add_or, err=self.err[1], prec=self.prec[0], add_full=1, true_or=self.true_or)
            x = self.pool(x)
        else:
            output = []
            for i in range(4):
                x1 = self.conv2(x, add_or = self.add_or, err=self.err[1]-bit_off, prec=self.prec[0], add_full=1, true_or=self.true_or)
                output.append(x1)
    #         x = self.pool(x)
            x = utils_own.avgpool2d_convert(output, size=2, config=config)
        x = F.relu(x)
        
        print("2", torch.sqrt(torch.mean(x**2)))
        print("3.weight", torch.sqrt(torch.mean(self.conv3.weight**2)))
        
        if config == "Normal":
            x = self.conv3(x, add_or = self.add_or, err=self.err[2], prec=self.prec[0], add_full=1, true_or=self.true_or)
            x = self.pool(x)
        else:
            output = []
            for i in range(4):
                x1 = self.conv3(x, add_or = self.add_or, err=self.err[2]-bit_off, prec=self.prec[0], add_full=1, true_or=self.true_or)
                output.append(x1)
            x = utils_own.avgpool2d_convert(output, size=2, config=config)
        x = F.relu(x)
        
#         print("3", torch.sqrt(torch.mean(x**2)))
#         print("4.weight", torch.sqrt(torch.mean(self.fc1.weight**2)))
        
        x = x.view(-1, 4*4*64)
        x = self.fc1(x, add_or = self.add_or, err=self.err[3], prec=self.prec[0], add_full=1, true_or=self.true_or)
#         print("4", torch.sqrt(torch.mean(x**2)))
#         print("\n")
        x = self.bn4(x)
        return x#, x1, x2, x3
    

class CONV_tiny(nn.Module):
    def __init__(self, add_or=True, max_pool=False):
        super(CONV_tiny, self).__init__()
   

        self.add_or = add_or
        self.register_buffer('conv1_result', torch.zeros(256,32,32,32))
        self.register_buffer('conv2_result', torch.zeros(256,32,16,16))
        self.register_buffer('conv3_result', torch.zeros(256,64,8,8))
        self.register_buffer('fc1_result', torch.zeros(256,10))
        
        self.conv1 = utils_class.Conv2d_Or(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.Conv2d_Or(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_class.Conv2d_Or(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1 = utils_class.Linear_Or(4*4*64, 10, bias=False)
        
        self.fcx1 = nn.Linear(32,10,bias=False)
        self.fcx2 = nn.Linear(32,10,bias=False)
        
        self.tanh = nn.Hardtanh(-1, 1)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        if max_pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.AvgPool2d(2)
            
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(10, affine=False)
        
        self.err = [7,7,7,7]
        self.prec = [7,7,7,7]
        self.true_or = False

                             
    def forward(self, x):
        input_file = os.path.join(dump_dir, 'input.npy')
        conv1_file = os.path.join(dump_dir, 'conv1_out.npy')
        conv1_pool_file = os.path.join(dump_dir, 'conv1_pool.npy')
        conv1_relu_file = os.path.join(dump_dir, 'conv1_relu.npy')
        
        conv2_file = os.path.join(dump_dir, 'conv2_out.npy')
        conv2_pool_file = os.path.join(dump_dir, 'conv2_pool.npy')
        conv2_relu_file = os.path.join(dump_dir, 'conv2_relu.npy')
        
        conv3_file = os.path.join(dump_dir, 'conv3_out.npy')
        conv3_pool_file = os.path.join(dump_dir, 'conv3_pool.npy')
        conv3_relu_file = os.path.join(dump_dir, 'conv3_relu.npy')
        
        fc1_file = os.path.join(dump_dir, 'fc1_out.npy')
        bn_file = os.path.join(dump_dir, 'bn_out.npy')
        
        conv1_weight = os.path.join(dump_dir, 'conv1_weight.npy')
        conv2_weight = os.path.join(dump_dir, 'conv2_weight.npy')
        conv3_weight = os.path.join(dump_dir, 'conv3_weight.npy')
        fc1_weight = os.path.join(dump_dir, 'fc1_weight.npy')
        bn_mean = os.path.join(dump_dir, 'bn_mean.npy')
        bn_var = os.path.join(dump_dir, 'bn_var.npy')
        
        if dump_param:
            conv1_w = (self.conv1.weight.data*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            np.save(conv1_weight, conv1_w)
            conv2_w = (self.conv2.weight.data*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            np.save(conv2_weight, conv2_w)
            conv3_w = (self.conv3.weight.data*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            np.save(conv3_weight, conv3_w)
            fc1_w = (self.fc1.weight.data*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            np.save(fc1_weight, fc1_w)
            bn_m = self.bn4.running_mean.cpu().numpy()
            np.save(bn_mean, bn_m)
            bn_v = self.bn4.running_var.cpu().numpy()
            np.save(bn_var, bn_v)
        
        if global_dump:
            try:
                input_v = np.load(input_file)
            except:
                input_v = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            else:
                input_c = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
                input_v = np.concatenate((input_c, input_v), axis=0)
            np.save(input_file, input_v)
        x = self.conv1(x, add_or = self.add_or, err=self.err[0], prec=self.prec[0], add_full=1, true_or=self.true_or)
        x = self.tanh(x)
        if global_dump:
            try:
                conv1_v = np.load(conv1_file)
            except:
                conv1_v = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            else:
                conv1_c = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
                conv1_v = np.concatenate((conv1_c, conv1_v), axis=0)
            np.save(conv1_file, conv1_v)
#         x = self.pool(x)
        if global_dump:
            try:
                conv1_pool_v = np.load(conv1_pool_file)
            except:
                conv1_pool_v = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            else:
                conv1_pool_c = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
                conv1_pool_v = np.concatenate((conv1_pool_c, conv1_pool_v), axis=0)
            np.save(conv1_pool_file, conv1_pool_v)
#         x = F.relu(x)
        if global_dump:
            try:
                conv1_relu_v = np.load(conv1_relu_file)
            except:
                conv1_relu_v = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            else:
                conv1_relu_c = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
                conv1_relu_v = np.concatenate((conv1_relu_c, conv1_relu_v), axis=0)
            np.save(conv1_relu_file, conv1_relu_v)
#         x = F.relu(self.pool(x))
        x = self.pool(F.relu(x))
        
        x = self.conv2(x, add_or = self.add_or, err=self.err[1], prec=self.prec[0], add_full=1, true_or=self.true_or)
        if global_dump:
            try:
                conv2_v = np.load(conv2_file)
            except:
                conv2_v = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            else:
                conv2_c = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
                conv2_v = np.concatenate((conv2_c, conv2_v), axis=0)
            np.save(conv2_file, conv2_v)
        x = self.tanh(x)
#         x = self.pool(x)
        if global_dump:
            try:
                conv2_pool_v = np.load(conv2_pool_file)
            except:
                conv2_pool_v = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            else:
                conv2_pool_c = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
                conv2_pool_v = np.concatenate((conv2_pool_c, conv2_pool_v), axis=0)
            np.save(conv2_pool_file, conv2_pool_v)
#         x = F.relu(x)
        if global_dump:
            try:
                conv2_relu_v = np.load(conv2_relu_file)
            except:
                conv2_relu_v = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            else:
                conv2_relu_c = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
                conv2_relu_v = np.concatenate((conv2_relu_c, conv2_relu_v), axis=0)
            np.save(conv2_relu_file, conv2_relu_v)
#         x = F.relu(self.pool(x))
        x = self.pool(F.relu(x))
        
        x = self.conv3(x, add_or = self.add_or, err=self.err[2], prec=self.prec[0], add_full=1, true_or=self.true_or)
        if global_dump:
            try:
                conv3_v = np.load(conv3_file)
            except:
                conv3_v = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            else:
                conv3_c = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
                conv3_v = np.concatenate((conv3_c, conv3_v), axis=0)
            np.save(conv3_file, conv3_v)
        x = self.tanh(x)
#         x = self.pool(x)
        if global_dump:
            try:
                conv3_pool_v = np.load(conv3_pool_file)
            except:
                conv3_pool_v = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            else:
                conv3_pool_c = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
                conv3_pool_v = np.concatenate((conv3_pool_c, conv3_pool_v), axis=0)
            np.save(conv3_pool_file, conv3_pool_v)
#         x = F.relu(x)
        if global_dump:
            try:
                conv3_relu_v = np.load(conv3_relu_file)
            except:
                conv3_relu_v = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            else:
                conv3_relu_c = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
                conv3_relu_v = np.concatenate((conv3_relu_c, conv3_relu_v), axis=0)
            np.save(conv3_relu_file, conv3_relu_v)
#         x = F.relu(self.pool(x))
        x = self.pool(F.relu(x))
        
        x = x.view(-1, 4*4*64)
        
        x = self.fc1(x, add_or = self.add_or, err=self.err[3], prec=self.prec[0], add_full=1, true_or=self.true_or)
        if global_dump:
            try:
                fc1_v = np.load(fc1_file)
            except:
                fc1_v = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
            else:
                fc1_c = ((x.detach())*128).ceil().clamp(-127,127).to(torch.int8).cpu().numpy()
                fc1_v = np.concatenate((fc1_c, fc1_v), axis=0)
            np.save(fc1_file, fc1_v)
        x = self.bn4(x)
        if global_dump:
            try:
                bn_v = np.load(bn_file)
            except:
                bn_v = x.detach().cpu().numpy()
            else:
                bn_c = x.detach().cpu().numpy()
                bn_v = np.concatenate((bn_c, bn_v), axis=0)
            np.save(bn_file, bn_v)
        return x#, x1, x2, x3
    
class CONV_tiny_and(nn.Module):
    def __init__(self, add_or=True):
        super(CONV_tiny_and, self).__init__()
        
        self.add_or = add_or
        
        self.conv1 = utils_own.Conv2d_And(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_own.Conv2d_And(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_own.Conv2d_And(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1 = utils_own.Linear_And(4*4*64, 10, bias=False)
        
        self.tanh = nn.Hardtanh()
        self.softmax = nn.LogSoftmax(dim=-1)
        
        self.pool = nn.AvgPool2d(2)
            
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(10, affine=False)
        
        self.err = [7,7,7,7]
        self.prec = [7,7,7,7]
        self.true_or = False

                             
    def forward(self, x):
        
        x = self.conv1(x, sc = self.add_or, err=self.err[0])
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.pool(F.relu(x))
        
        x = self.conv2(x, sc = self.add_or, err=self.err[1])
        x = self.bn2(x)
        x = self.tanh(x)
        x = self.pool(F.relu(x))
        
        x = self.conv3(x, sc = self.add_or, err=self.err[2])
        x = self.bn3(x)
        x = self.tanh(x)
        x = self.pool(F.relu(x))
        
        x = x.view(-1, 4*4*64)
        x = self.fc1(x, sc = self.add_or, err=self.err[3])
        x = self.bn4(x)
        return x#, x1, x2, x3
    
class CONV_tiny_xnor(nn.Module):
    def __init__(self, add_or=True, max_pool=False):
        super(CONV_tiny_xnor, self).__init__()
   

        self.add_or = add_or
        
        self.conv1 = utils_own.Conv2d_Xnor(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_own.Conv2d_Xnor(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_own.Conv2d_Xnor(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1 = utils_own.Linear_Xnor(4*4*64, 10, bias=False)
        
        self.fcx1 = nn.Linear(32,10,bias=False)
        self.fcx2 = nn.Linear(32,10,bias=False)
        
        self.tanh = nn.Hardtanh()
        self.softmax = nn.LogSoftmax(dim=-1)
        
        if max_pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.AvgPool2d(2)
            
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(10, affine=False)
        
        self.err = [7,7,7,7]
        self.prec = [7,7,7,7]
        self.true_or = False

                             
    def forward(self, x):
        
        x = self.conv1(x, xnor = self.add_or, err=self.err[0])
        x = self.tanh(x)
        x = self.pool(F.relu(x))
        
        x = self.conv2(x, xnor = self.add_or, err=self.err[1])
        x = self.tanh(x)
        x = self.pool(F.relu(x))
        
        x = self.conv3(x, xnor = self.add_or, err=self.err[2])
        x = self.tanh(x)
        x = self.pool(F.relu(x))
        
        x = x.view(-1, 4*4*64)
        x = self.fc1(x, xnor = self.add_or, err=self.err[3])
        x = self.bn4(x)
        return x#, x1, x2, x3
    
def conv3(in_plane, out_plane):
    return utils_class.Conv2d_Add_Partial(in_plane, out_plane, kernel_size=3, padding=1, bias=False)
def conv5(in_plane, out_plane):
    return utils_class.Conv2d_Add_Partial(in_plane, out_plane, kernel_size=5, padding=2, bias=False)
    
class CONV_tiny_add_nas(nn.Module):
    def __init__(self, add_or=True, max_pool=False):
        super(CONV_tiny_add_nas, self).__init__()
        self.add_or = add_or
        
        self.choose1 = utils_class.Binary_Select()
        self.choose2 = utils_class.Binary_Select()
        self.choose3 = utils_class.Binary_Select()
        
        self.conv1_00 = conv5(3, 32)
        self.conv2_00 = conv5(32, 32)
        self.conv3_00 = conv5(32, 64)
        
        self.conv1_10 = conv3(3, 32)
        self.conv1_11 = conv3(32, 32)
        self.conv2_10 = conv3(32, 32)
        self.conv2_11 = conv3(32, 32)
        self.conv3_10 = conv3(32, 32)
        self.conv3_11 = conv3(32, 64)
        
        self.fc1 = utils_class.Conv2d_Add_Partial(64, 10, kernel_size=4, padding=0, bias=False)
        self.tanh = nn.Hardtanh(0,1)
        self.pool = nn.AvgPool2d(2)
        
        self.bn1_00 = utils_class.BatchNorm2d_fixed(32)
        self.bn2_00 = utils_class.BatchNorm2d_fixed(32)
        self.bn3_00 = utils_class.BatchNorm2d_fixed(64)
        
        self.bn1_10 = utils_class.BatchNorm2d_fixed(32)
        self.bn1_11 = utils_class.BatchNorm2d_fixed(32)
        self.bn2_10 = utils_class.BatchNorm2d_fixed(32)
        self.bn2_11 = utils_class.BatchNorm2d_fixed(32)
        self.bn3_10 = utils_class.BatchNorm2d_fixed(32)
        self.bn3_11 = utils_class.BatchNorm2d_fixed(64)
        
        self.bn4 = utils_class.BatchNorm1d_fixed(10, affine=False)
        
        self.err = [7,7,7,7]
        self.true_or = False
        
    def conv_forward(self, x, conv, bn, err, pool=False):
        x = conv(x, err)
        if pool:
            x = self.pool(x)
        x = bn(x)
        x = self.tanh(x)
        return x
        
    def forward(self, x):
        # Branch 0
        x_0 = self.conv_forward(x, self.conv1_00, self.bn1_00, self.err[0], pool=True)
        
        # Branch 1
        x_1 = self.conv_forward(x, self.conv1_10, self.bn1_10, self.err[0])
        x_1 = self.conv_forward(x_1, self.conv1_11, self.bn1_11, self.err[0], pool=True)
        
        x = self.choose1(x_0, x_1)
       
        # Branch 0
        x_0 = self.conv_forward(x, self.conv2_00, self.bn2_00, self.err[1], pool=True)
        
        # Branch 1
        x_1 = self.conv_forward(x, self.conv2_10, self.bn2_10, self.err[1])
        x_1 = self.conv_forward(x_1, self.conv2_11, self.bn2_11, self.err[1], pool=True)
        
        x = self.choose2(x_0, x_1)
        
        # Branch 0
        x_0 = self.conv_forward(x, self.conv3_00, self.bn3_00, self.err[2], pool=True)
        
        # Branch 1
        x_1 = self.conv_forward(x, self.conv3_10, self.bn3_10, self.err[2])
        x_1 = self.conv_forward(x_1, self.conv3_11, self.bn3_11, self.err[2], pool=True)
        
        x = self.choose3(x_0, x_1)
        
        x = self.fc1(x, err=self.err[3])
        x = x.view(-1, 10)
        
        x = self.bn4(x)
        return x
    
class CONV_tiny_add_group(nn.Module):
    def __init__(self, add_or=True, max_pool=False):
        super(CONV_tiny_add_group, self).__init__()
        self.add_or = add_or
        self.register_buffer('conv1_result', torch.zeros(256,32,32,32))
        self.register_buffer('conv2_result', torch.zeros(256,32,16,16))
        self.register_buffer('conv3_result', torch.zeros(256,64,8,8))
        self.register_buffer('fc1_result', torch.zeros(256,10))
        
        self.conv1 = utils_class.Conv2d_Add_Partial(3, 64, kernel_size=5, padding=2, bias=False)
        
        self.conv2_1 = utils_class.Conv2d_Add_Partial(8, 8, kernel_size=5, padding=2, bias=False)
        self.conv2_2 = utils_class.Conv2d_Add_Partial(8, 8, kernel_size=5, padding=2, bias=False)
        self.conv2_3 = utils_class.Conv2d_Add_Partial(8, 8, kernel_size=5, padding=2, bias=False)
        self.conv2_4 = utils_class.Conv2d_Add_Partial(8, 8, kernel_size=5, padding=2, bias=False)
        self.conv2_5 = utils_class.Conv2d_Add_Partial(8, 8, kernel_size=5, padding=2, bias=False)
        self.conv2_6 = utils_class.Conv2d_Add_Partial(8, 8, kernel_size=5, padding=2, bias=False)
        self.conv2_7 = utils_class.Conv2d_Add_Partial(8, 8, kernel_size=5, padding=2, bias=False)
        self.conv2_8 = utils_class.Conv2d_Add_Partial(8, 8, kernel_size=5, padding=2, bias=False)
        
        self.conv3_1 = utils_class.Conv2d_Add_Partial(8, 16, kernel_size=5, padding=2, bias=False)
        self.conv3_2 = utils_class.Conv2d_Add_Partial(8, 16, kernel_size=5, padding=2, bias=False)
        self.conv3_3 = utils_class.Conv2d_Add_Partial(8, 16, kernel_size=5, padding=2, bias=False)
        self.conv3_4 = utils_class.Conv2d_Add_Partial(8, 16, kernel_size=5, padding=2, bias=False)
        self.conv3_5 = utils_class.Conv2d_Add_Partial(8, 16, kernel_size=5, padding=2, bias=False)
        self.conv3_6 = utils_class.Conv2d_Add_Partial(8, 16, kernel_size=5, padding=2, bias=False)
        self.conv3_7 = utils_class.Conv2d_Add_Partial(8, 16, kernel_size=5, padding=2, bias=False)
        self.conv3_8 = utils_class.Conv2d_Add_Partial(8, 16, kernel_size=5, padding=2, bias=False)
        
        self.fc1 = utils_class.Conv2d_Add_Partial(128, 10, kernel_size=4, padding=0, bias=False)
        
        self.tanh = nn.Hardtanh(0,1)
        if max_pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.AvgPool2d(2)
            
        self.bn1 = utils_class.BatchNorm2d_fixed(64)
        self.bn2 = utils_class.BatchNorm2d_fixed(64)
        self.bn3 = utils_class.BatchNorm2d_fixed(128)
        self.bn4 = utils_class.BatchNorm1d_fixed(10, affine=False)
        
        self.err = [7,7,7,7]
        self.true_or = False
    
    def forward(self, x):
        x = self.conv1(x, err=self.err[0])
#         x = F.relu(x)
        x = self.pool(x)
        x = self.bn1(x)
        x = self.tanh(x)
        
        x_1 = self.conv2_1(x[:,:8].clone(), err=self.err[1])
        x_2 = self.conv2_2(x[:,8:16].clone(), err=self.err[1])
        x_3 = self.conv2_3(x[:,16:24].clone(), err=self.err[1])
        x_4 = self.conv2_4(x[:,24:32].clone(), err=self.err[1])
        x_5 = self.conv2_5(x[:,32:40].clone(), err=self.err[1])
        x_6 = self.conv2_6(x[:,40:48].clone(), err=self.err[1])
        x_7 = self.conv2_7(x[:,48:56].clone(), err=self.err[1])
        x_8 = self.conv2_8(x[:,56:64].clone(), err=self.err[1])
        x = torch.cat([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8], dim=1)
        x = self.pool(x)
        x = self.bn2(x)
        x = self.tanh(x)
        
        x_1 = torch.cat([x[:,0:1], x[:,8:9],   x[:,16:17], x[:,24:25], x[:,32:33], x[:,40:41], x[:,48:49], x[:,56:57]], dim=1).clone()
        x_2 = torch.cat([x[:,1:2], x[:,9:10],  x[:,17:18], x[:,25:26], x[:,33:34], x[:,41:42], x[:,49:50], x[:,57:58]], dim=1).clone()
        x_3 = torch.cat([x[:,2:3], x[:,10:11], x[:,18:19], x[:,26:27], x[:,34:35], x[:,42:43], x[:,50:51], x[:,58:59]], dim=1).clone()
        x_4 = torch.cat([x[:,3:4], x[:,11:12], x[:,19:20], x[:,27:28], x[:,35:36], x[:,43:44], x[:,51:52], x[:,59:60]], dim=1).clone()
        x_5 = torch.cat([x[:,4:5], x[:,12:13], x[:,20:21], x[:,28:29], x[:,36:37], x[:,44:45], x[:,52:53], x[:,60:61]], dim=1).clone()
        x_6 = torch.cat([x[:,5:6], x[:,13:14], x[:,21:22], x[:,29:30], x[:,37:38], x[:,45:46], x[:,53:54], x[:,61:62]], dim=1).clone()
        x_7 = torch.cat([x[:,6:7], x[:,14:15], x[:,22:23], x[:,30:31], x[:,38:39], x[:,46:47], x[:,54:55], x[:,62:63]], dim=1).clone()
        x_8 = torch.cat([x[:,7:8], x[:,15:16], x[:,23:24], x[:,31:32], x[:,39:40], x[:,47:48], x[:,55:56], x[:,63:64]], dim=1).clone()
        x_1 = self.conv3_1(x_1, err=self.err[2])
        x_2 = self.conv3_2(x_2, err=self.err[2])
        x_3 = self.conv3_3(x_3, err=self.err[2])
        x_4 = self.conv3_4(x_4, err=self.err[2])
        x_5 = self.conv3_5(x_5, err=self.err[2])
        x_6 = self.conv3_6(x_6, err=self.err[2])
        x_7 = self.conv3_7(x_7, err=self.err[2])
        x_8 = self.conv3_8(x_8, err=self.err[2])
        x = torch.cat([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8], dim=1)
        x = self.pool(x)
        x = self.bn3(x)
        x = self.tanh(x)
        
        x = self.fc1(x, err=self.err[3])
        x = x.view(-1, 10)
        x = self.bn4(x)
        return x
    
class CONV_tiny_approx(nn.Module):
    def __init__(self, err_prof):
        super(CONV_tiny_approx, self).__init__()
        self.register_buffer('err_prof', torch.zeros(128,128))
        self.conv1 = utils_class.Conv2d_Approx(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.Conv2d_Approx(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_class.Conv2d_Approx(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1   = utils_class.Conv2d_Approx(64, 10, kernel_size=4, padding=0, bias=False)
        
        self.pool = nn.MaxPool2d(2)
        self.tanh = nn.Hardtanh(0,1)
    def forward(self, x):
        x = F.relu(self.conv1(x, self.err_prof))
        x = self.tanh(self.pool(x))
        np.save("conv1_res", x.data.cpu().numpy())
        x = F.relu(self.conv2(x, self.err_prof))
        x = self.tanh(self.pool(x))
        np.save("conv2_res", x.data.cpu().numpy())
        x = F.relu(self.conv3(x, self.err_prof))
        x = self.tanh(self.pool(x))
        x = self.fc1(x, self.err_prof)
        x = x.view(-1,10)
        return x
        
class CONV_tiny_bin(nn.Module):
    def __init__(self):
        super(CONV_tiny_bin, self).__init__()
        self.conv1 = utils_class.BinarizeConv2d(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.BinarizeConv2d(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_class.BinarizeConv2d(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1   = utils_class.BinarizeConv2d(64, 10, kernel_size=4, padding=0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(10, affine=False)
        
        self.tanh = nn.Hardtanh(0,1)
        self.pool = nn.AvgPool2d(2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.fc1(x)
        x = x.view(-1,10)
        x = self.bn4(x)
        return x
    
class MLP_minimal(nn.Module):
    def __init__(self):
        super(MLP_minimal, self).__init__()
        self.fc1 = utils_class.Linear_Or(784, 128, bias=False)
        self.fc2 = utils_class.Linear_Or(128, 10, bias=False)
        self.bn1 = utils_class.BatchNorm1d_fixed(10, affine=False)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x, err=5)
        x = F.relu(x)
        x = self.fc2(x, err=5)
        return x
    
class CONV_minimal_add_partial_v2(nn.Module):
    def __init__(self):
        super(CONV_minimal_add_partial_v2, self).__init__()
        
        self.bn1 = utils_class.BatchNorm2d_fixed(6)
        self.bn2 = utils_class.BatchNorm2d_fixed(16)
        self.bn3 = utils_class.BatchNorm1d_fixed(10, affine=False)
        
        
        self.conv1 = utils_class.Conv2d_Add_Partial(1, 6, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.Conv2d_Add_Partial(6, 16, kernel_size=5, padding=0, bias=False)
        self.fc1   = utils_class.Conv2d_Add_Partial(16, 10, kernel_size=5, padding=0, bias=False)
        
        self.tanh = nn.Hardtanh(0,1)
        self.pool = nn.AvgPool2d(2)
                             
    def forward(self, x):
        x = self.conv1(x, 7)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.tanh(x)
        
        x = self.conv2(x, 7)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.tanh(x)
        
        x = self.fc1(x, 7)
        x = x.view(-1,10)
        x = self.bn3(x)
        return x
    
class VGG16_add_partial(nn.Module):
    def __init__(self, uniform=False, sc_compute='1d_bin', generator='lfsr', legacy=False, half_pool=False):
        super(VGG16_add_partial, self).__init__()
        self.conv1 = utils_class.Conv2d_Add_Partial(3, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = utils_class.Conv2d_Add_Partial(64, 64, kernel_size=3, padding=1, bias=False)
        self.drop1 = nn.Dropout2d()
        self.conv3 = utils_class.Conv2d_Add_Partial(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv4 = utils_class.Conv2d_Add_Partial(128, 128, kernel_size=3, padding=1, bias=False)
        self.drop2 = nn.Dropout2d()
        self.conv5 = utils_class.Conv2d_Add_Partial(128, 256, kernel_size=3, padding=1, bias=False)
        self.conv6 = utils_class.Conv2d_Add_Partial(256, 256, kernel_size=3, padding=1, bias=False)
        self.conv7 = utils_class.Conv2d_Add_Partial(256, 256, kernel_size=3, padding=1, bias=False)
        self.drop3 = nn.Dropout2d()
        self.conv8 = utils_class.Conv2d_Add_Partial(256, 512, kernel_size=3, padding=1, bias=False)
        self.conv9 = utils_class.Conv2d_Add_Partial(512, 512, kernel_size=3, padding=1, bias=False)
        self.conv10 = utils_class.Conv2d_Add_Partial(512, 512, kernel_size=3, padding=1, bias=False)
        self.drop4 = nn.Dropout2d()
        self.conv11 = utils_class.Conv2d_Add_Partial(512, 512, kernel_size=3, padding=1, bias=False)
        self.conv12 = utils_class.Conv2d_Add_Partial(512, 512, kernel_size=3, padding=1, bias=False)
        self.conv13 = utils_class.Conv2d_Add_Partial(512, 512, kernel_size=3, padding=1, bias=False)
        self.drop5 = nn.Dropout2d()
        self.fc1 = utils_class.Conv2d_Add_Partial(512, 10, kernel_size=1, padding=0, bias=False)
        self.pool = nn.AvgPool2d(2)
        self.bn1 = utils_class.BatchNorm2d_fixed(64)
        self.bn2 = utils_class.BatchNorm2d_fixed(64)
        self.bn3 = utils_class.BatchNorm2d_fixed(128)
        self.bn4 = utils_class.BatchNorm2d_fixed(128)
        self.bn5 = utils_class.BatchNorm2d_fixed(256)
        self.bn6 = utils_class.BatchNorm2d_fixed(256)
        self.bn7 = utils_class.BatchNorm2d_fixed(256)
        self.bn8 = utils_class.BatchNorm2d_fixed(512)
        self.bn9 = utils_class.BatchNorm2d_fixed(512)
        self.bn10 = utils_class.BatchNorm2d_fixed(512)
        self.bn11 = utils_class.BatchNorm2d_fixed(512)
        self.bn12 = utils_class.BatchNorm2d_fixed(512)
        self.bn13 = utils_class.BatchNorm2d_fixed(512)
        self.bnfc1 = utils_class.BatchNorm1d_fixed(10)
        self.tanh = nn.Hardtanh(0,1)

        self.err = [7,7,7,7,7,7]
        self.z_unit = [5,5,5,5,5,5]
        self.load_unit = [2,2,2,2,2,2]
        self.load_wait_w = [2,2,2,2,2,2]
        self.load_wait_a = [2,2,2,2,2,2]
        self.generator = generator
        self.compute = sc_compute
        self.legacy = legacy
        self.half_pool = half_pool
        
        if np.min(self.err)<4:
            uniform=True
        if uniform:
            for mod in self.modules():
                if isinstance(mod, nn.Conv2d):
                    mod.weight.data.uniform_()
                    mod.weight_org.uniform_()
    def forward(self, x):
        if self.half_pool:
            pool_offset=1
        else:
            pool_offset=0
        x = self.conv1(x, err=self.err[0], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait_w[0], load_wait_a=self.load_wait_a[0])
        err_0 = max(self.err[0]-pool_offset, 5)
        x = self.tanh(self.bn1(x))
        x = self.conv2(x, err=err_0, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait_w[0], load_wait_a=self.load_wait_a[0])
        x = self.tanh(self.pool(self.bn2(x)))
        x = self.drop1(x)
        
        x = self.conv3(x, err=self.err[1], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[1], legacy=self.legacy, load_unit=self.load_unit[1], load_wait_w=self.load_wait_w[1], load_wait_a=self.load_wait_a[1])
        err_1 = max(self.err[1]-pool_offset, 5)
        x = self.tanh(self.bn3(x))
        x = self.conv4(x, err=err_1, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[1], legacy=self.legacy, load_unit=self.load_unit[1], load_wait_w=self.load_wait_w[1], load_wait_a=self.load_wait_a[1])
        x = self.tanh(self.pool(self.bn4(x)))
        x = self.drop2(x)
        
        x = self.conv5(x, err=self.err[2], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[2], legacy=self.legacy, load_unit=self.load_unit[2], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[2])
        err_2 = max(self.err[2]-pool_offset, 5)
        x = self.tanh(self.bn5(x))
        x = self.conv6(x, err=self.err[2], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[2], legacy=self.legacy, load_unit=self.load_unit[2], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[2])
        x = self.tanh(self.bn6(x))
        x = self.conv7(x, err=err_2, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[2], legacy=self.legacy, load_unit=self.load_unit[2], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[2])
        x = self.tanh(self.pool(self.bn7(x)))
        x = self.drop3(x)
        
        x = self.conv8(x, err=self.err[3], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[3], legacy=self.legacy, load_unit=self.load_unit[3], load_wait_w=self.load_wait_w[3], load_wait_a=self.load_wait_a[3])
        err_3 = max(self.err[3]-pool_offset, 5)
        x = self.tanh(self.bn8(x))
        x = self.conv9(x, err=self.err[3], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[3], legacy=self.legacy, load_unit=self.load_unit[3], load_wait_w=self.load_wait_w[3], load_wait_a=self.load_wait_a[3])
        x = self.tanh(self.bn9(x))
        x = self.conv10(x, err=err_3, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[3], legacy=self.legacy, load_unit=self.load_unit[3], load_wait_w=self.load_wait_w[3], load_wait_a=self.load_wait_a[3])
        x = self.tanh(self.pool(self.bn10(x)))
        x = self.drop4(x)
        
        x = self.conv11(x, err=self.err[4], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[4], legacy=self.legacy, load_unit=self.load_unit[4], load_wait_w=self.load_wait_w[4], load_wait_a=self.load_wait_a[4])
        err_4 = max(self.err[4]-pool_offset, 5)
        x = self.tanh(self.bn11(x))
        x = self.conv12(x, err=self.err[4], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[4], legacy=self.legacy, load_unit=self.load_unit[4], load_wait_w=self.load_wait_w[4], load_wait_a=self.load_wait_a[4])
        x = self.tanh(self.bn12(x))
        x = self.conv13(x, err=err_4, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[4], legacy=self.legacy, load_unit=self.load_unit[4], load_wait_w=self.load_wait_w[4], load_wait_a=self.load_wait_a[4])
        x = self.tanh(self.pool(self.bn13(x)))
        x = self.drop5(x)
        
        x = self.fc1(x, err=self.err[5], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[5], legacy=self.legacy, load_unit=self.load_unit[5], load_wait_w=self.load_wait_w[5], load_wait_a=self.load_wait_a[5])
        x = x.view(-1,10)
        x = self.bnfc1(x)
        return x
        

class CONV_tiny_mult_run(nn.Module):
    def __init__(self, add_or=True, max_pool=False, num_classes=10, c_ins=3, uniform=False, generator='lfsr',
                sc_compute='1d_bin', legacy=False, long_s=False, thres=1., larger_t=False):
        super(CONV_tiny_mult_run, self).__init__()
        self.add_or = add_or
        self.register_buffer('conv1_result', torch.zeros(256,32,32,32))
        self.register_buffer('conv2_result', torch.zeros(256,32,16,16))
        self.register_buffer('conv3_result', torch.zeros(256,64,8,8))
        self.register_buffer('fc1_result', torch.zeros(256,10))
        
        # Short version
        self.bn1 = utils_class.BatchNorm2d_fixed(32)
        self.bn2 = utils_class.BatchNorm2d_fixed(32)
        self.bn3 = utils_class.BatchNorm2d_fixed(64)
        self.bn4 = utils_class.BatchNorm1d_fixed(num_classes)
        
        self.conv1 = utils_class.Conv2d_Add_Partial(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.Conv2d_Add_Partial(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_class.Conv2d_Add_Partial(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1   = utils_class.Conv2d_Add_Partial(64, num_classes, kernel_size=4, padding=0, bias=False)
        
        # Long version
        self.bn1_1 = utils_class.BatchNorm2d_fixed(32)
        self.bn2_1 = utils_class.BatchNorm2d_fixed(32)
        self.bn3_1 = utils_class.BatchNorm2d_fixed(64)
        self.bn4_1 = utils_class.BatchNorm1d_fixed(num_classes)
        self.fc1_1 = utils_class.Conv2d_Add_Partial(64, num_classes, kernel_size=4, padding=0, bias=False)
        
        self.conv1_l = utils_class.Conv2d_Add_Partial(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2_l = utils_class.Conv2d_Add_Partial(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3_l = utils_class.Conv2d_Add_Partial(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1_l   = utils_class.Conv2d_Add_Partial(64, num_classes, kernel_size=4, padding=0, bias=False)
        
        self.tanh = nn.Hardtanh(0,1)
        
        if max_pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.AvgPool2d(2)
        
        self.err = [7,7,7,7]
        self.z_unit = [3,3,4,4]
        self.load_unit = [2,2,2,2]
        self.load_wait_w = [1,2,4,4]
        self.load_wait_a = [5,5,5,5]
        self.true_or = False
        self.dump = False
        self.dump_dir = ""
        self.num_classes = num_classes
        self.generator = generator
        self.compute = sc_compute
        self.legacy = legacy
        self.long_s = long_s
        self.thres = thres
        self.larger_t = larger_t
        if uniform:
            self.conv1.weight_org *= 2
            self.conv2.weight_org *= 2
            self.conv3.weight_org *= 2
            self.conv1.weight.data*= 2
            self.conv2.weight.data*= 2
            self.conv3.weight.data*= 2
                             
    def forward(self, x_in, target=None):
        size_total = 0
        size_long = 0
        if self.long_s:
            conv1_l = self.conv1_l
            conv2_l = self.conv2_l
            conv3_l = self.conv3_l
            fc1_l = self.fc1_l
        else:
            conv1_l = self.conv1
            conv2_l = self.conv2
            conv3_l = self.conv3
            fc1_l = self.fc1
        x = self.conv1(x_in, err=self.err[0], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait_w[0], load_wait_a=self.load_wait_a[0])
        y = conv1_l(x_in, err=7, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait_w[0], load_wait_a=self.load_wait_a[0])
        long_ind = ((x>self.thres) + (x<-self.thres)).float().detach()
        size_total += torch.prod(torch.tensor(x.size()))
        size_long += long_ind.sum()
        if self.larger_t:
            x = x*long_ind + y*(1-long_ind)
        else:
            x = x*(1-long_ind) + y*long_ind
        if global_userelu:
            x = F.relu(x)
        x = self.pool(x)
        x = self.bn1(x)
        x = self.tanh(x)

        x_in = x
        x = self.conv2(x_in, err=self.err[1], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[1], legacy=self.legacy, load_unit=self.load_unit[1], load_wait_w=self.load_wait_w[1], load_wait_a=self.load_wait_a[1])
        y = conv2_l(x_in, err=7, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[1], legacy=self.legacy, load_unit=self.load_unit[1], load_wait_w=self.load_wait_w[1], load_wait_a=self.load_wait_a[1])
        long_ind = ((x>self.thres) + (x<-self.thres)).float().detach()
        size_total += torch.prod(torch.tensor(x.size()))
        size_long += long_ind.sum()
        if self.larger_t:
            x = x*long_ind + y*(1-long_ind)
        else:
            x = x*(1-long_ind) + y*long_ind
        if global_userelu:
            x = F.relu(x)
        x = self.pool(x)
        x = self.bn2(x)
        x = self.tanh(x)

        x_in = x
        x = self.conv3(x_in, err=self.err[2], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[2], legacy=self.legacy, load_unit=self.load_unit[2], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[2])
        y = conv3_l(x_in, err=7, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[2], legacy=self.legacy, load_unit=self.load_unit[2], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[2])
        long_ind = ((x>self.thres) + (x<-self.thres)).float().detach()
        size_total += torch.prod(torch.tensor(x.size()))
        size_long += long_ind.sum()
        if self.larger_t:
            x = x*long_ind + y*(1-long_ind)
        else:
            x = x*(1-long_ind) + y*long_ind
        if global_userelu:
            x = F.relu(x)
        x = self.pool(x)
        x = self.bn3(x)
        x = self.tanh(x)
        
        x = self.fc1(x, err=self.err[3], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[3], legacy=self.legacy, load_unit=self.load_unit[3], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[3])
        x = x.view(-1, self.num_classes)
        x = self.bn4(x)
        print(size_long / size_total)
        return x

class CONV_tiny_nas(nn.Module):
    def __init__(self, add_or=True, max_pool=False, num_classes=10, c_ins=3, uniform=False, generator='lfsr',
                sc_compute='1d_bin', legacy=False):
        super(CONV_tiny_nas, self).__init__()
        self.add_or = add_or
        self.register_buffer('conv1_result', torch.zeros(256,32,32,32))
        self.register_buffer('conv2_result', torch.zeros(256,32,16,16))
        self.register_buffer('conv3_result', torch.zeros(256,64,8,8))
        self.register_buffer('fc1_result', torch.zeros(256,10))
        
        # NAS weights
        self.register_parameter('sel1', nn.Parameter(torch.ones(4)))
        self.register_parameter('sel2', nn.Parameter(torch.ones(4)))
        self.register_parameter('sel3', nn.Parameter(torch.ones(4)))
        
        self.register_buffer('size1', torch.tensor([2.4576, 4.9152, 9.8304, 19.6608]))
        self.register_buffer('size2', torch.tensor([6.5536, 13.1072, 26.2144, 52.4288]))
        self.register_buffer('size3', torch.tensor([3.2768, 6.5536, 13.1072, 26.2144]))
        
        # 16-bit version
        self.bn1_16 = utils_class.BatchNorm2d_fixed(32)
        self.bn2_16 = utils_class.BatchNorm2d_fixed(32)
        self.bn3_16 = utils_class.BatchNorm2d_fixed(64)
        self.bn4 = utils_class.BatchNorm1d_fixed(num_classes)
        
        self.conv1_16 = utils_class.Conv2d_Add_Partial(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2_16 = utils_class.Conv2d_Add_Partial(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3_16 = utils_class.Conv2d_Add_Partial(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1   = utils_class.Conv2d_Add_Partial(64, num_classes, kernel_size=4, padding=0, bias=False)
        
        # 32-bit version
        self.bn1_32 = utils_class.BatchNorm2d_fixed(32)
        self.bn2_32 = utils_class.BatchNorm2d_fixed(32)
        self.bn3_32 = utils_class.BatchNorm2d_fixed(64)
        
        self.conv1_32 = utils_class.Conv2d_Add_Partial(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2_32 = utils_class.Conv2d_Add_Partial(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3_32 = utils_class.Conv2d_Add_Partial(32, 64, kernel_size=5, padding=2, bias=False)
        
        # 64-bit version
        self.bn1_64 = utils_class.BatchNorm2d_fixed(32)
        self.bn2_64 = utils_class.BatchNorm2d_fixed(32)
        self.bn3_64 = utils_class.BatchNorm2d_fixed(64)
        
        self.conv1_64 = utils_class.Conv2d_Add_Partial(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2_64 = utils_class.Conv2d_Add_Partial(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3_64 = utils_class.Conv2d_Add_Partial(32, 64, kernel_size=5, padding=2, bias=False)
        
        # 128-bit version
        self.bn1_128 = utils_class.BatchNorm2d_fixed(32)
        self.bn2_128 = utils_class.BatchNorm2d_fixed(32)
        self.bn3_128 = utils_class.BatchNorm2d_fixed(64)
        
        self.conv1_128 = utils_class.Conv2d_Add_Partial(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2_128 = utils_class.Conv2d_Add_Partial(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3_128 = utils_class.Conv2d_Add_Partial(32, 64, kernel_size=5, padding=2, bias=False)
        
        self.tanh = nn.Hardtanh(0,1)
        
        if max_pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.AvgPool2d(2)
        
        self.err = [7,7,7,7]
        self.z_unit = [3,3,4,4]
        self.load_unit = [2,2,2,2]
        self.load_wait_w = [1,2,4,4]
        self.load_wait_a = [5,5,5,5]
        self.true_or = False
        self.dump = False
        self.dump_dir = ""
        self.num_classes = num_classes
        self.generator = generator
        self.compute = sc_compute
        self.legacy = legacy
        if uniform:
            self.conv1_16.weight_org *= 2
            self.conv2_16.weight_org *= 2
            self.conv3_16.weight_org *= 2
            self.conv1_16.weight.data*= 2
            self.conv2_16.weight.data*= 2
            self.conv3_16.weight.data*= 2
            self.conv1_32.weight_org *= 2
            self.conv2_32.weight_org *= 2
            self.conv3_32.weight_org *= 2
            self.conv1_32.weight.data*= 2
            self.conv2_32.weight.data*= 2
            self.conv3_32.weight.data*= 2
                             
    def forward(self, x_in):
        dump = False
        cost = 0
        x = []
        prec_range = [4,5,6,7]
        module_list = [self.conv1_16, self.conv1_32, self.conv1_64, self.conv1_128]
        bn_list = [self.bn1_16, self.bn1_32, self.bn1_64, self.bn1_128]
        for i,module in enumerate(module_list):
            x_cur = module(x_in, err=prec_range[i], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait_w[0], load_wait_a=self.load_wait_a[0])
            if global_userelu:
                x_cur = F.relu(x_cur)
            x_cur = self.pool(x_cur)
            x_cur = bn_list[i](x_cur)
            x_cur = self.tanh(x_cur)
            x.append(x_cur)
        x_comb = torch.stack(x, dim=-1)
        sel_norm = F.softmax(self.sel1, dim=-1)
        if self.training:
            sel_raw = torch.rand([]).to(x_in.device)
            sel_pred = 0
            comp_cur = sel_norm.data[0]
            while sel_raw>=comp_cur:
                sel_pred += 1
                comp_cur = comp_cur + sel_norm.data[sel_pred]
        else:
            _, sel_pred = sel_norm.topk(1,0,True,True)
            sel_pred = sel_pred.item()
        x = (x_comb * sel_norm).sum(-1)
        cost = cost + torch.dot(sel_norm, self.size1)
        x.data = x_comb.data[...,sel_pred]
        if dump:
            if self.training:
                print(sel_raw, sel_pred, sel_norm)
            else:
                print(sel_pred, sel_norm)
        x_in = x
        x = []
        module_list = [self.conv2_16, self.conv2_32, self.conv2_64, self.conv2_128]
        bn_list = [self.bn2_16, self.bn2_32, self.bn2_64, self.bn2_128]
        for i,module in enumerate(module_list):
            x_cur = module(x_in, err=prec_range[i], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait_w[0], load_wait_a=self.load_wait_a[0])
            if global_userelu:
                x_cur = F.relu(x_cur)
            x_cur = self.pool(x_cur)
            x_cur = bn_list[i](x_cur)
            x_cur = self.tanh(x_cur)
            x.append(x_cur)
        x_comb = torch.stack(x, dim=-1)
        sel_norm = F.softmax(self.sel2, dim=-1)
        if self.training:
            sel_raw = torch.rand([]).to(x_in.device)
            sel_pred = 0
            comp_cur = sel_norm.data[0]
            while sel_raw>=comp_cur:
                sel_pred += 1
                comp_cur = comp_cur + sel_norm.data[sel_pred]
        else:
            _, sel_pred = sel_norm.topk(1,0,True,True)
            sel_pred = sel_pred.item()
        x = (x_comb * sel_norm).sum(-1)
        cost = cost + torch.dot(sel_norm, self.size1)
        x.data = x_comb.data[...,sel_pred]
        if dump:
            if self.training:
                print(sel_raw, sel_pred, sel_norm)
            else:
                print(sel_pred, sel_norm)
            
        x_in = x
        x = []
        module_list = [self.conv3_16, self.conv3_32, self.conv3_64, self.conv3_128]
        bn_list = [self.bn3_16, self.bn3_32, self.bn3_64, self.bn3_128]
        for i,module in enumerate(module_list):
            x_cur = module(x_in, err=prec_range[i], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait_w[0], load_wait_a=self.load_wait_a[0])
            if global_userelu:
                x_cur = F.relu(x_cur)
            x_cur = self.pool(x_cur)
            x_cur = bn_list[i](x_cur)
            x_cur = self.tanh(x_cur)
            x.append(x_cur)
        x_comb = torch.stack(x, dim=-1)
        sel_norm = F.softmax(self.sel3, dim=-1)
        if self.training:
            sel_raw = torch.rand([]).to(x_in.device)
            sel_pred = 0
            comp_cur = sel_norm.data[0]
            while sel_raw>=comp_cur:
                sel_pred += 1
                comp_cur = comp_cur + sel_norm.data[sel_pred]
        else:
            _, sel_pred = sel_norm.topk(1,0,True,True)
            sel_pred = sel_pred.item()
        x = (x_comb * sel_norm).sum(-1)
        cost = cost + torch.dot(sel_norm, self.size1)
        x.data = x_comb.data[...,sel_pred]
        if dump:
            if self.training:
                print(sel_raw, sel_pred, sel_norm)
            else:
                print(sel_pred, sel_norm)
        
        x = self.fc1(x, err=self.err[3], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[3], legacy=self.legacy, load_unit=self.load_unit[3], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[3])
        x = x.view(-1, self.num_classes)
        x = self.bn4(x)
        if dump:
            print(self.sel1, self.sel2, self.sel3)
        return x, cost
    
class CONV_tiny_mult_pass(nn.Module):
    def __init__(self, add_or=True, max_pool=False, num_classes=10, c_ins=3, uniform=False, generator='lfsr',
                sc_compute='1d_bin', legacy=False, pred_pos=3, ensemble=False, long_s=False):
        super(CONV_tiny_mult_pass, self).__init__()
        self.add_or = add_or
        self.register_buffer('conv1_result', torch.zeros(256,32,32,32))
        self.register_buffer('conv2_result', torch.zeros(256,32,16,16))
        self.register_buffer('conv3_result', torch.zeros(256,64,8,8))
        self.register_buffer('fc1_result', torch.zeros(256,10))
        
        # Short version
        self.bn1 = utils_class.BatchNorm2d_fixed(32)
        self.bn2 = utils_class.BatchNorm2d_fixed(32)
        self.bn3 = utils_class.BatchNorm2d_fixed(64)
        self.bn4 = utils_class.BatchNorm1d_fixed(num_classes)
        
        self.conv1 = utils_class.Conv2d_Add_Partial(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.Conv2d_Add_Partial(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_class.Conv2d_Add_Partial(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1   = utils_class.Conv2d_Add_Partial(64, num_classes, kernel_size=4, padding=0, bias=False)
        
        # Predictor
        self.conv1_pred = utils_class.Conv2d_Add_Partial(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2_pred = utils_class.Conv2d_Add_Partial(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3_pred = utils_class.Conv2d_Add_Partial(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc_pred = utils_class.Conv2d_Add_Partial(64, 2, kernel_size=4, padding=0, bias=False)
        
        self.bn1_pred = utils_class.BatchNorm2d_fixed(32)
        self.bn2_pred = utils_class.BatchNorm2d_fixed(32)
        self.bn3_pred = utils_class.BatchNorm2d_fixed(64)
        self.bn_pred = utils_class.BatchNorm1d_fixed(2)
        
        # Long version
        self.bn1_1 = utils_class.BatchNorm2d_fixed(32)
        self.bn2_1 = utils_class.BatchNorm2d_fixed(32)
        self.bn3_1 = utils_class.BatchNorm2d_fixed(64)
        self.bn4_1 = utils_class.BatchNorm1d_fixed(num_classes)
        self.fc1_1 = utils_class.Conv2d_Add_Partial(64, num_classes, kernel_size=4, padding=0, bias=False)
        
        self.conv1_l = utils_class.Conv2d_Add_Partial(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2_l = utils_class.Conv2d_Add_Partial(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3_l = utils_class.Conv2d_Add_Partial(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1_l   = utils_class.Conv2d_Add_Partial(64, num_classes, kernel_size=4, padding=0, bias=False)
        
        self.tanh = nn.Hardtanh(0,1)
        
        if max_pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.AvgPool2d(2)
        
        self.err = [7,7,7,7]
        self.z_unit = [3,3,4,4]
        self.load_unit = [2,2,2,2]
        self.load_wait_w = [1,2,4,4]
        self.load_wait_a = [5,5,5,5]
        self.true_or = False
        self.dump = False
        self.dump_dir = ""
        self.num_classes = num_classes
        self.generator = generator
        self.compute = sc_compute
        self.legacy = legacy
        self.pred_pos = pred_pos
        self.ensemble = ensemble
        self.long_s = long_s
        if uniform:
            self.conv1.weight_org *= 2
            self.conv2.weight_org *= 2
            self.conv3.weight_org *= 2
            self.conv1.weight.data*= 2
            self.conv2.weight.data*= 2
            self.conv3.weight.data*= 2
            self.conv1_pred.weight_org *= 2
            self.conv2_pred.weight_org *= 2
            self.conv3_pred.weight_org *= 2
            self.conv1_pred.weight.data*= 2
            self.conv2_pred.weight.data*= 2
            self.conv3_pred.weight.data*= 2
                             
    def forward(self, x_in, target=None):
        if self.long_s:
            conv1_l = self.conv1_l
            conv2_l = self.conv2_l
            conv3_l = self.conv3_l
            fc1_l = self.fc1_l
        else:
            conv1_l = self.conv1
            conv2_l = self.conv2
            conv3_l = self.conv3
            fc1_l = self.fc1
        x = self.conv1(x_in, err=self.err[0], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait_w[0], load_wait_a=self.load_wait_a[0])
        y = conv1_l(x_in, err=7, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait_w[0], load_wait_a=self.load_wait_a[0])
        x_pred = self.conv1_pred(x_in, err=self.err[0], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait_w[0], load_wait_a=self.load_wait_a[0])
        if global_userelu:
            x = F.relu(x)
            y = F.relu(y)
            x_pred = F.relu(x_pred)
        x = self.pool(x)
        x = self.bn1(x)
        x = self.tanh(x)
        y = self.pool(y)
        y = self.bn1_1(y)
        y = self.tanh(y)
        x_pred = self.pool(x_pred)
        x_pred = self.bn1_pred(x_pred)
        x_pred = self.tanh(x_pred)

        if self.pred_pos==1:
            x_pred = x.data.clone().detach()
        x = self.conv2(x, err=self.err[1], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[1], legacy=self.legacy, load_unit=self.load_unit[1], load_wait_w=self.load_wait_w[1], load_wait_a=self.load_wait_a[1])
        y = conv2_l(y, err=7, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[1], legacy=self.legacy, load_unit=self.load_unit[1], load_wait_w=self.load_wait_w[1], load_wait_a=self.load_wait_a[1])
        x_pred = self.conv2_pred(x_pred, err=self.err[1], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[1], legacy=self.legacy, load_unit=self.load_unit[1], load_wait_w=self.load_wait_w[1], load_wait_a=self.load_wait_a[1])
        if global_userelu:
            x = F.relu(x)
            y = F.relu(y)
            x_pred = F.relu(x_pred)
        x = self.pool(x)
        x = self.bn2(x)
        x = self.tanh(x)
        y = self.pool(y)
        y = self.bn2_1(y)
        y = self.tanh(y)
        x_pred = self.pool(x_pred)
        x_pred = self.bn2_pred(x_pred)
        x_pred = self.tanh(x_pred)

        if self.pred_pos==2:
            x_pred = x.data.clone().detach()
        x = self.conv3(x, err=self.err[2], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[2], legacy=self.legacy, load_unit=self.load_unit[2], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[2])
        y = conv3_l(y, err=7, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[2], legacy=self.legacy, load_unit=self.load_unit[2], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[2])
        x_pred = self.conv3_pred(x_pred, err=self.err[2], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[2], legacy=self.legacy, load_unit=self.load_unit[2], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[2])
        if global_userelu:
            x = F.relu(x)
            y = F.relu(y)
            x_pred = F.relu(x_pred)
        x = self.pool(x)
        x = self.bn3(x)
        x = self.tanh(x)
        y = self.pool(y)
        y = self.bn3_1(y)
        y = self.tanh(y)
        x_pred = self.pool(x_pred)
        x_pred = self.bn3_pred(x_pred)
        x_pred = self.tanh(x_pred)

        if self.pred_pos==3:
            x_pred = x.data.clone().detach()
        x_pred = self.fc_pred(x_pred, err=self.err[3], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[3], legacy=self.legacy, load_unit=self.load_unit[3], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[3])
#         x_b = self.fc_pred(x_b)
        x_pred = x_pred.view(-1, 2)
        x_pred = self.bn_pred(x_pred)
        
        x = self.fc1(x, err=self.err[3], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[3], legacy=self.legacy, load_unit=self.load_unit[3], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[3])
        x = x.view(-1, self.num_classes)
        x = self.bn4(x)
        y = fc1_l(y, err=7, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[3], legacy=self.legacy, load_unit=self.load_unit[3], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[3])
        y = y.view(-1, self.num_classes)
        y = self.bn4_1(y)
        if self.ensemble:
            # x_pred predicts 1 if consistent, meaning the second one is consistent
            y = x*0.2+ y*0.8
        x = torch.cat((x, x_pred, y), dim=-1)
        return x
    
class CONV_tiny_add_partial(nn.Module):
    def __init__(self, add_or=True, max_pool=False, num_classes=10, c_ins=3, uniform=False, generator='lfsr',
                sc_compute='1d_bin', legacy=False, relu=False):
        super(CONV_tiny_add_partial, self).__init__()
        self.add_or = add_or
        self.register_buffer('conv1_result', torch.zeros(256,32,32,32))
        self.register_buffer('conv2_result', torch.zeros(256,32,16,16))
        self.register_buffer('conv3_result', torch.zeros(256,64,8,8))
        self.register_buffer('fc1_result', torch.zeros(256,10))
        
#         if global_usebn:
        self.bn1 = utils_class.BatchNorm2d_fixed(32)
        self.bn2 = utils_class.BatchNorm2d_fixed(32)
        self.bn3 = utils_class.BatchNorm2d_fixed(64)
#         else:
#             self.bn1 = nn.Identity()
#             self.bn2 = nn.Identity()
#             self.bn3 = nn.Identity()
        self.bn4 = utils_class.BatchNorm1d_fixed(num_classes)
#         self.bn4 = utils_class.BatchNorm1d_fixed(num_classes, affine=False)
        
        self.bn1_1 = utils_class.BatchNorm2d_fixed(32)
        self.bn2_1 = utils_class.BatchNorm2d_fixed(32)
        self.bn3_1 = utils_class.BatchNorm2d_fixed(64)
        self.bn4_1 = utils_class.BatchNorm1d_fixed(num_classes, affine=False)
        
        self.conv1 = utils_class.Conv2d_Add_Partial(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.Conv2d_Add_Partial(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_class.Conv2d_Add_Partial(32, 64, kernel_size=5, padding=2, bias=False)
#         self.fc1 = utils_class.Linear_Or(64, num_classes, bias=False)
#         self.fc1_1 = utils_class.Linear_Or(64, num_classes, bias=False)
        self.fc1   = utils_class.Conv2d_Add_Partial(64, num_classes, kernel_size=4, padding=0, bias=False)
        self.fc1_1 = utils_class.Conv2d_Add_Partial(64, num_classes, kernel_size=4, padding=0, bias=False)
        
#         self.tunable = nn.Module()
#         self.tunable.bn1_1 = self.bn1_1
#         self.tunable.bn2_1 = self.bn2_1
#         self.tunable.bn3_1 = self.bn3_1
#         self.tunable.bn4_1 = self.bn4_1
#         self.tunable.fc1_1 = self.fc1_1
#         self.tunable.bn1 = self.bn1
#         self.tunable.bn2 = self.bn2
#         self.tunable.bn3 = self.bn3
#         self.tunable.bn4 = self.bn4
#         self.tunable.fc1 = self.fc1
#         self.tunable = [self.bn1_1, self.bn2_1, self.bn3_1, self.bn4_1]
        
        self.tanh = nn.Hardtanh(0,1)
        
        if max_pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.AvgPool2d(2)

#         self.bn1 = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.bn4 = nn.BatchNorm1d(10, affine=False)
        
        self.err = [7,7,7,7]
        self.z_unit = [3,3,4,4]
        self.load_unit = [2,2,2,2]
        self.load_wait_w = [1,2,4,4]
        self.load_wait_a = [5,5,5,5]
        self.true_or = False
        self.dump = False
        self.dump_dir = ""
        self.mult_pass = False
        self.num_classes = num_classes
        self.generator = generator
        self.compute = sc_compute
        self.legacy = legacy
        self.relu = relu
        if uniform:
            self.conv1.weight_org *= 2
            self.conv2.weight_org *= 2
            self.conv1.weight.data*= 2
            self.conv2.weight.data*= 2
#             self.conv1.weight_org.uniform_(-1,1)
#             self.conv2.weight_org.uniform_(-1,1)
#             self.conv3.weight_org.uniform_(-1,1)
#             self.fc1.weight_org.uniform_(-1,1)
#             self.conv1.weight.data.uniform_(-1,1)
#             self.conv2.weight.data.uniform_(-1,1)
#             self.conv3.weight.data.uniform_(-1,1)
#             self.fc1.weight.data.uniform_(-1,1)
                             
    def forward(self, x_in, target=None):
#         print(self.relu)
        x = self.conv1(x_in, err=self.err[0], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait_w[0], load_wait_a=self.load_wait_a[0])
        x = self.bn1(x)
        if self.relu:
            x = F.relu(x)
        x = self.pool(x)
#         x = self.bn1(x)
        x = self.tanh(x)

#             x = self.conv2(x, err=err_cur[1], bn=self.bn2)
        x = self.conv2(x, err=self.err[1], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[1], legacy=self.legacy, load_unit=self.load_unit[1], load_wait_w=self.load_wait_w[1], load_wait_a=self.load_wait_a[1])
        x = self.bn2(x)
        if self.relu:
            x = F.relu(x)
        x = self.pool(x)
#         x = self.bn2(x)
        x = self.tanh(x)

#             x = self.conv3(x, err=err_cur[2], bn=self.bn3)
        x = self.conv3(x, err=self.err[2], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[2], legacy=self.legacy, load_unit=self.load_unit[2], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[2])
        x = self.bn3(x)
        if self.relu:
            x = F.relu(x)
        x = self.pool(x)
#         x = self.bn3(x)
        x = self.tanh(x)

        x = self.fc1(x, err=self.err[3], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[3], legacy=self.legacy, load_unit=self.load_unit[3], load_wait_w=self.load_wait_w[2], load_wait_a=self.load_wait_a[3])
        x = x.view(-1, self.num_classes)
        x = self.bn4(x)
        return x
        

class CONV_middle_add(nn.Module):
    def __init__(self, add_or=True, max_pool=False, expand=1, classes=10):
        super(CONV_middle_add, self).__init__()
        self.add_or = add_or
        self.expand = expand
        
        self.conv1 = utils_class.Conv2d_Add(3, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = utils_class.Conv2d_Add(32,32, kernel_size=3, padding=1, bias=False)
        self.conv3 = utils_class.Conv2d_Add(32,64, kernel_size=3, padding=1, bias=False)
        self.conv4 = utils_class.Conv2d_Add(64,64, kernel_size=3, padding=1, bias=False)
        self.conv5 = utils_class.Conv2d_Add(64,128,kernel_size=3, padding=1, bias=False)
        self.conv6 = utils_class.Conv2d_Add(128,128,kernel_size=3,padding=1, bias=False)
        self.fc1 = utils_class.Linear_Add(4*4*128, classes, bias=False)
        
        self.tanh = nn.Hardtanh()
        
        self.pool = nn.AvgPool2d(2)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn = nn.BatchNorm1d(classes, affine=False)
        
    def forward(self, x):
        x = self.tanh(F.relu(self.bn1(self.conv1(x))))
        x = self.tanh(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.tanh(F.relu(self.bn3(self.conv3(x))))
        x = self.tanh(self.pool(F.relu(self.bn4(self.conv4(x)))))
        x = self.tanh(F.relu(self.bn5(self.conv5(x))))
        x = self.tanh(self.pool(F.relu(self.bn6(self.conv6(x)))))
        
        x = x.view(-1, 4*4*128)
        x = self.bn(self.fc1(x))
        return x

        
class CONV_tiny_add(nn.Module):
    def __init__(self, add_or=True, max_pool=False, expand=1, classes=10):
        super(CONV_tiny_add, self).__init__()
   

        self.add_or = add_or
        self.expand = expand
        self.register_buffer('conv1_result', torch.zeros(256,32,32,32))
        self.register_buffer('conv2_result', torch.zeros(256,32,16,16))
        self.register_buffer('conv3_result', torch.zeros(256,64,8,8))
        self.register_buffer('fc1_result', torch.zeros(256,10))
        
        self.conv1 = utils_class.Conv2d_Add(3, 32*expand, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.Conv2d_Add(32*expand, 32*expand, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_class.Conv2d_Add(32*expand, 64*expand, kernel_size=5, padding=2, bias=False)
        self.fc1 = utils_class.Linear_Add(4*4*64*expand, classes, bias=False)
        
        self.tanh = nn.Hardtanh()
        
        if max_pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.AvgPool2d(2)
            
        self.bn1 = nn.BatchNorm2d(32*expand, affine=False)
        self.bn2 = nn.BatchNorm2d(32*expand, affine=False)
        self.bn3 = nn.BatchNorm2d(64*expand, affine=False)
        self.bn4 = nn.BatchNorm1d(classes, affine=False)
        
        self.err = [7,7,7,7]
        self.true_or = False

                             
    def forward(self, x):
        x = self.conv1(x, err=self.err[0])
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.tanh(x)
        
        x = self.conv2(x, err=self.err[1])
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.tanh(x)
        
        x = self.conv3(x, err=self.err[2])
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.tanh(x)
        
        x = x.view(-1, 4*4*64*self.expand)
        x = self.fc1(x, err=self.err[3])
        x = self.bn4(x)
        return x
    
class CONV_minimal_xnor(nn.Module):
    def __init__(self):
        super(CONV_minimal_xnor, self).__init__()
        
        self.conv1 = utils_own.Conv2d_Xnor(1, 6, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_own.Conv2d_Xnor(6, 16, kernel_size=5, padding=0, bias=False)
        self.fc1 = utils_own.Linear_Xnor(5*5*16, 120, bias=False)
        self.fc2 = utils_own.Linear_Xnor(120, 84, bias=False)
        self.fc3 = utils_own.Linear_Xnor(84, 10, bias=False)
        self.bn = nn.BatchNorm1d(10, affine=False)
        
        self.tanh = nn.Hardtanh()
        self.pool = nn.AvgPool2d(2)
        self.add_or = False
        self.err = 7
        
    def forward(self, x):
        x = F.relu(self.conv1(x, xnor=self.add_or, err=self.err))
        x = self.tanh(x)
        x = self.pool(x)
        
        x = F.relu(self.conv2(x, xnor=self.add_or, err=self.err))
        x = self.tanh(x)
        x = self.pool(x)
        x = x.view(-1, 16*5*5)
        
        x = F.relu(self.fc1(x, xnor=self.add_or, err=self.err))
        x = self.tanh(x)
        x = F.relu(self.fc2(x, xnor=self.add_or, err=self.err))
        x = self.tanh(x)
        x = self.fc3(x, xnor=self.add_or, err=self.err)
        x = self.bn(x)
        return x

class CONV_minimal_pool(nn.Module):
    def __init__(self, add_or=True, max_pool=False, mux=False):
        super(CONV_minimal_pool, self).__init__()
        
        self.conv1 = utils_own.Conv2d_Or(1, 6, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_own.Conv2d_Or(6, 16, kernel_size=5, padding=0, bias=False)
        self.fc1 = utils_own.Linear_Or(5*5*16, 120, bias=False)
        self.fc2 = utils_own.Linear_Or(120, 84, bias=False)
        self.fc3 = utils_own.Linear_Or(84, 10, bias=False)
        self.bn = nn.BatchNorm1d(10, affine=False)
        
        self.pool = nn.AvgPool2d(2)
        self.add_or = True
        self.err=16
    def forward(self, x):
        x = self.pool(self.conv1(x, add_or = self.add_or, err=self.err-2, prec=self.err))
        self.conv1_result = x.data.clone()
        x = F.relu(x)

        x = self.pool(self.conv2(x, add_or = self.add_or, err=self.err-2, prec=self.err))
        self.conv2_result = x.data.clone()
        x = F.relu(x)
#         print(x.size())
        x = x.view(-1, 16*5*5)

        x = F.relu(self.fc1(x, add_or = self.add_or, err=self.err, prec=self.err))
        x = F.relu(self.fc2(x, add_or = self.add_or, err=self.err, prec=self.err))
        x = self.fc3(x, add_or = self.add_or, err=self.err, prec=self.err)
        self.fc1_result = x.data.clone()   
        x = self.bn(x)
        return x
    
class CONV_minimal_stream_pool(nn.Module):
    def __init__(self):
        super(CONV_minimal_stream_pool, self).__init__()
        
        self.conv1 = utils_class.Conv2d_Or_Streamout_relu_pool(1, 6, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.Conv2d_Or_Stream_relu_pool(6, 16, kernel_size=5, padding=0, bias=False)
        self.fc1 = utils_class.Linear_Or_Stream(5*5*16, 120, bias=False)
        self.fc2 = utils_class.Linear_Or_Stream(120, 84, bias=False)
        self.fc3 = utils_class.Linear_Or_Stream(84, 10, bias=False)
        self.bn = nn.BatchNorm1d(10, affine=False)
        
        self.tanh = nn.Hardtanh(-0.5, 0.5)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        self.pool = nn.AvgPool2d(2)
    
    def forward(self, x):
        x = self.conv1(x, err=self.err)
        x,x_v = self.conv2(x)
        x = x.view(x.size(0)*x.size(1), x.size(2), 5*5*16)
        x_v = x_v.view(-1, 5*5*16)
        x_pos, x_neg = self.fc1((x,x_v))
        x = utils_functional.subtract_relu(x_pos, x_neg)
        
        x_pos, x_neg = self.fc2(x)
        x = utils_functional.subtract_relu(x_pos, x_neg)
        
        x = self.fc3(x)
        x = x[0][1] - x[1][1]
        x = self.bn(x)
        return x
        
    
class CONV_minimal_stream(nn.Module):
    def __init__(self):
        super(CONV_minimal_stream, self).__init__()
        
        self.conv1 = utils_class.Conv2d_Or_Streamout(1, 6, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.Conv2d_Or_Stream(6, 16, kernel_size=5, padding=0, bias=False)
        self.fc1 = utils_class.Linear_Or_Stream(5*5*16, 120, bias=False)
        self.fc2 = utils_class.Linear_Or_Stream(120, 84, bias=False)
        self.fc3 = utils_class.Linear_Or_Stream(84, 10, bias=False)
        self.bn = nn.BatchNorm1d(10, affine=False)
        
        self.tanh = nn.Hardtanh(-0.5, 0.5)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        self.pool = nn.AvgPool2d(2)
        
    def forward(self, x):
        x_pos, x_neg = self.conv1(x, err=self.err)
        x = utils_functional.subtract_relu(x_pos, x_neg)
        x = utils_functional.avgpool2d_stream(x)

        x_pos, x_neg = self.conv2(x)
        x = utils_functional.subtract_relu(x_pos, x_neg)
        x, x_v = utils_functional.avgpool2d_stream(x)

        x = x.view(x.size(0), x.size(1), 5*5*16)
        x_v = x_v.view(-1, 5*5*16)
        x_pos, x_neg = self.fc1((x,x_v))
        x = utils_functional.subtract_relu(x_pos, x_neg)
        
        x_pos, x_neg = self.fc2(x)
        x = utils_functional.subtract_relu(x_pos, x_neg)
        
        x = self.fc3(x)
        x = x[0][1] - x[1][1]
        x = self.bn(x)
        return x
    
class CONV_minimal_add_partial(nn.Module):
    def __init__(self, uniform=False, legacy=False, generator='lfsr', sc_compute='1d_bin'):
        super(CONV_minimal_add_partial, self).__init__()
        self.register_buffer('conv1_result', None)
        self.register_buffer('conv2_result', None)
        self.register_buffer('fc1_result', None)
        
        if global_usebn:
            self.bn1 = utils_class.BatchNorm2d_fixed(6)
            self.bn2 = utils_class.BatchNorm2d_fixed(16)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
        
        self.conv1 = utils_class.Conv2d_Add_Partial(1, 6, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.Conv2d_Add_Partial(6, 16, kernel_size=5, padding=0, bias=False)
        self.fc1 = utils_class.Linear_Or(5*5*16, 120, bias=False)
        self.fc2 = utils_class.Linear_Or(120, 84, bias=False)
        self.fc3 = utils_class.Linear_Or(84, 10, bias=False)
        self.bn = utils_class.BatchNorm1d_fixed(10, affine=False)
        
        self.pool = nn.AvgPool2d(2)
        self.tanh = nn.Hardtanh(0,1)
        self.generator = generator
        self.compute = sc_compute
        if uniform:
            for mod in self.modules():
                mod.weight.data.uniform_(-1,1)
                mod.weight_org.uniform_(-1,1)
        self.err = 5
        self.legacy = legacy
    def forward(self, x):
        x = self.conv1(x, err=self.err, forward=self.compute, generator=self.generator, legacy=self.legacy)
#         x = F.relu(x)
        x = self.pool(x)
        x = self.bn1(x)
        x = self.tanh(x)
        
        x = self.conv2(x, err=self.err, forward=self.compute, generator=self.generator, legacy=self.legacy)
#         x = F.relu(x)
        x = self.pool(x)
        x = self.bn2(x)
        x = self.tanh(x)
        
        x = x.view(-1, 16*5*5)
        x = self.tanh(self.fc1(x, err=self.err, generator=self.generator))
        x = self.tanh(self.fc2(x, err=self.err, generator=self.generator))
        x = self.fc3(x, err=self.err, generator=self.generator)
        x = self.bn(x)
        return x
        
        
        
class CONV_minimal(nn.Module):
    def __init__(self, add_or=True, max_pool=False, mux=False):
        super(CONV_minimal, self).__init__()
        
        self.mux = mux
        self.add_or = add_or
        self.register_buffer('conv1_result', None)
        self.register_buffer('conv2_result', None)
        self.register_buffer('fc1_result', None)
        
        self.conv1 = utils_class.Conv2d_Or(1, 6, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.Conv2d_Or(6, 16, kernel_size=5, padding=0, bias=False)
        self.fc1 = utils_class.Linear_Or(5*5*16, 120, bias=False)
        self.fc2 = utils_class.Linear_Or(120, 84, bias=False)
        self.fc3 = utils_class.Linear_Or(84, 10, bias=False)
        self.bn = nn.BatchNorm1d(10, affine=False)
        
        if max_pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.AvgPool2d(2)
        self.err=16
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x, add_or = self.add_or, err=self.err, prec=self.err)))
        self.conv1_result = x.data.clone()
#         x = F.relu(x)

        x = self.pool(F.relu(self.conv2(x, add_or = self.add_or, err=self.err, prec=self.err)))
        self.conv2_result = x.data.clone()
#         x = F.relu(x)
#         print(x.size())
        x = x.view(-1, 16*5*5)

        x = F.relu(self.fc1(x, add_or = self.add_or, err=self.err, prec=self.err))
        x = F.relu(self.fc2(x, add_or = self.add_or, err=self.err, prec=self.err))
        x = self.fc3(x, add_or = self.add_or, err=self.err, prec=self.err)
        self.fc1_result = x.data.clone()   
        x = self.bn(x)
        return x
    
class AlexNet_small_ref(nn.Module):
    def __init__(self, dropout = 0.7, maxpool=False, bn=False, initialization=None):
        super(AlexNet_small_ref, self).__init__()
        
        if maxpool:
            self.pool = nn.MaxPool2d(2,2)
        else:
            self.pool = nn.AvgPool2d(2,2)
            
        self.do_bn = True
        
        self.dropconv = nn.Dropout2d(p=0.3)
        self.dropfc = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv2d(3, 96, 7, padding=3)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2, groups=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1, groups=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1, groups=2)
        self.bn4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1, groups=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(8*8*256, 4096)
        self.bnfc1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bnfc2 = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, 200)
        self.softmax = nn.LogSoftmax(dim=1)
        
#         self.init_weights(initialization)
        
    def init_weights(self, initialization):
        if initialization=='xavier':
            for mod in self.modules():
                if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                    nn.init.xavier_normal_(mod.weight.data)
        if initialization=='he':
            for mod in self.modules():
                if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                    nn.init.kaiming_normal_(mod.weight.data)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        if self.do_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropconv(x)
        
        x = self.conv2(x)
        x = self.pool(x)
        if self.do_bn:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropconv(x)
        
        x = self.conv3(x)
        if self.do_bn:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropconv(x)
    
        x = F.relu(self.conv4(x))
        if self.do_bn:
            x = self.bn4(x)
        x = F.relu(x)
        x = self.dropconv(x)
    
        x = self.conv5(x)
        x = self.pool(x)
        if self.do_bn:
            x = self.bn5(x)
        x = F.relu(x)
        x = self.dropconv(x)
        
        x = x.view(-1, 8*8*256)
        x = self.fc1(x)
        if self.do_bn:
            x = self.bnfc1(x)
        x = F.relu(x)
        x = self.dropfc(x)

        x = self.fc2(x)
        if self.do_bn:
            x = self.bnfc2(x)
        x = F.relu(x)
        x = self.dropfc(x)
        
        x = self.fc3(x)
#         x = self.softmax(x)
        return x

class AlexNet_small(nn.Module):
    def __init__(self, dropout = 0, maxpool=False, bn=False, initialization=None, prec=8):
        super(AlexNet_small, self).__init__()
        
        if maxpool:
            self.pool = nn.MaxPool2d(2,2)
        else:
            self.pool = nn.AvgPool2d(2,2)
            
        self.do_bn = True
        
        self.tanh = nn.Hardtanh()
        self.dropconv = nn.Dropout2d(p=0)
        self.dropfc = nn.Dropout(p=dropout)
        self.conv1 = utils_own.Conv2d_Or(3, 96, 7, padding=3)
        self.conv2 = utils_own.Conv2d_Or(96, 256, 5, padding=2, groups=1)
        self.conv3 = utils_own.Conv2d_Or(256, 384, 3, padding=1, groups=1)
        self.conv4 = utils_own.Conv2d_Or(384, 384, 3, padding=1, groups=1)
        self.conv5 = utils_own.Conv2d_Or(384, 256, 3, padding=1, groups=1)

        self.fc1 = utils_own.Linear_Or(8*8*256, 4096)
        self.fc2 = utils_own.Linear_Or(4096, 4096)
        self.fc3 = utils_own.Linear_Or(4096, 200)
        self.bnfc3 = nn.BatchNorm1d(200, affine=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropconv(x)
        x = self.pool(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.dropconv(x)
        x = self.pool(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.dropconv(x)
        x = F.relu(x)
    
        x = self.conv4(x)
        x = self.dropconv(x)
        x = F.relu(x)
    
        x = self.conv5(x)
        x = self.dropconv(x)
        x = self.pool(x)
        x = F.relu(x)
        
        x = x.view(-1, 8*8*256)
        x = self.fc1(x)
        x = self.dropfc(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.dropfc(x)
        x = F.relu(x)
        
        x = self.fc3(x, relu=False)
        x = self.bnfc3(x)
        return x
    
class AlexNet_test(nn.Module):
    def __init__(self, max_pool=False, prec=7, full_out=False):
        super(AlexNet_test, self).__init__()

        self.conv1 = utils_own.Conv2d_Or(3, 64, kernel_size=11, padding=2, stride=4)
        self.conv2 = utils_own.Conv2d_Or(64, 192, kernel_size=5, padding=2, stride=1)
        self.conv3 = utils_own.Conv2d_Or(192, 384, kernel_size=3, padding=1, stride=1)
        self.conv4 = utils_own.Conv2d_Or(384, 256, kernel_size=3, padding=1, stride=1)
        self.conv5 = utils_own.Conv2d_Or(256, 256, kernel_size=3, padding=1, stride=1)
        if max_pool:
            self.pool = nn.MaxPool2d(3, stride=2)
        else:
            self.pool = nn.AvgPool2d(3, stride=2)
            
        self.err = 7
            
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.add_or = False
        self.prec = prec
        self.full_out = full_out
        self.relu = nn.ReLU()
        
        self.bn = nn.BatchNorm1d(1001, affine=False)
        
        self.fc1 = utils_own.Linear_Or(9216, 4096, bias=False)
        self.fc2 = utils_own.Linear_Or(4096, 4096, bias=False)
        self.fc3 = utils_own.Linear_Or(4096, 1001, bias=False)
        
    def forward(self, x, add_or=False, quantize=False):
        add_or = add_or or self.add_or
#         if prec is None:
#             prec = self.prec
        if quantize:
            prec = 8
        else:
            prec = None
        x = x/5
        x = self.relu(self.conv1(x, add_or=add_or, prec=prec, err=self.err))
        x = self.pool(x)
        x = self.relu(self.conv2(x, add_or=add_or, prec=prec, err=self.err))
        x = self.pool(x)
        x = self.relu(self.conv3(x, add_or=add_or, prec=prec, err=self.err))
        x = self.relu(self.conv4(x, add_or=add_or, prec=prec, err=self.err))
        x = self.relu(self.conv5(x, add_or=add_or, prec=prec, err=self.err))
        x = self.pool(x)
        
        x = x.view(-1, 256*6*6)
        x = self.relu(self.fc1(x, add_or=add_or, prec=prec, err=self.err))
        x = self.relu(self.fc2(x, add_or=add_or, prec=prec, err=self.err))
        x = self.fc3(x, add_or=add_or, prec=prec, err=self.err)
        x = self.bn(x)
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = utils_own.Conv2d_Or(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.AvgPool2d(stride)
        self.conv2 = utils_own.Conv2d_Or(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                utils_own.Conv2d_Or(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(stride)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = utils_own.Conv2d_Or(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc1 = utils_own.Linear_Or(512*block.expansion, num_classes)

        self.add_or = False
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        for mod in self.modules():
            if hasattr(mod, 'add_or'):
                mod.add_or = self.add_or
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

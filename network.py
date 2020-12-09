import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils_class

'''
File containing all model definitions
'''
    
class VGG16_add_partial(nn.Module):
    '''
    VGG16 modified for SC
    Stream length (err)/load unit/load wait values are specified using a string of 6 integers. Pooling layers automatically use half the stream length. Stream length = 2**{value specified}
    The first value is used for the layer1-2, second for layer3-4, third for layer5-7, fourth for layer8-10, fifth for layer11-13, sixth for FC layers
    E.g.: to achieve load unit=2, load wait=2, stream length=64 for all layers without pooling and 32 for all layers with pooling, err=[666666], load_unit=[222222], load_wait=[222222]
    '''
    def __init__(self, uniform=False, sc_compute='1d_bin', generator='lfsr', legacy=False, half_pool=True):
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
        self.load_wait = [2,2,2,2,2,2]
        self.generator = generator
        self.compute = sc_compute
        self.legacy = legacy
        self.half_pool = half_pool
        
        # Scale up weights for low precision and stream length. Other underflow prevents effective training
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
        x = self.conv1(x, err=self.err[0], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait[0], load_wait_a=self.load_wait[0])
        err_0 = max(self.err[0]-pool_offset, 5)
        x = self.tanh(self.bn1(x))
        x = self.conv2(x, err=err_0, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait[0], load_wait_a=self.load_wait[0])
        x = self.tanh(self.pool(self.bn2(x)))
        x = self.drop1(x)
        
        x = self.conv3(x, err=self.err[1], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[1], legacy=self.legacy, load_unit=self.load_unit[1], load_wait_w=self.load_wait[1], load_wait_a=self.load_wait[1])
        err_1 = max(self.err[1]-pool_offset, 5)
        x = self.tanh(self.bn3(x))
        x = self.conv4(x, err=err_1, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[1], legacy=self.legacy, load_unit=self.load_unit[1], load_wait_w=self.load_wait[1], load_wait_a=self.load_wait[1])
        x = self.tanh(self.pool(self.bn4(x)))
        x = self.drop2(x)
        
        x = self.conv5(x, err=self.err[2], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[2], legacy=self.legacy, load_unit=self.load_unit[2], load_wait_w=self.load_wait[2], load_wait_a=self.load_wait[2])
        err_2 = max(self.err[2]-pool_offset, 5)
        x = self.tanh(self.bn5(x))
        x = self.conv6(x, err=self.err[2], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[2], legacy=self.legacy, load_unit=self.load_unit[2], load_wait_w=self.load_wait[2], load_wait_a=self.load_wait[2])
        x = self.tanh(self.bn6(x))
        x = self.conv7(x, err=err_2, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[2], legacy=self.legacy, load_unit=self.load_unit[2], load_wait_w=self.load_wait[2], load_wait_a=self.load_wait[2])
        x = self.tanh(self.pool(self.bn7(x)))
        x = self.drop3(x)
        
        x = self.conv8(x, err=self.err[3], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[3], legacy=self.legacy, load_unit=self.load_unit[3], load_wait_w=self.load_wait[3], load_wait_a=self.load_wait[3])
        err_3 = max(self.err[3]-pool_offset, 5)
        x = self.tanh(self.bn8(x))
        x = self.conv9(x, err=self.err[3], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[3], legacy=self.legacy, load_unit=self.load_unit[3], load_wait_w=self.load_wait[3], load_wait_a=self.load_wait[3])
        x = self.tanh(self.bn9(x))
        x = self.conv10(x, err=err_3, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[3], legacy=self.legacy, load_unit=self.load_unit[3], load_wait_w=self.load_wait[3], load_wait_a=self.load_wait[3])
        x = self.tanh(self.pool(self.bn10(x)))
        x = self.drop4(x)
        
        x = self.conv11(x, err=self.err[4], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[4], legacy=self.legacy, load_unit=self.load_unit[4], load_wait_w=self.load_wait[4], load_wait_a=self.load_wait[4])
        err_4 = max(self.err[4]-pool_offset, 5)
        x = self.tanh(self.bn11(x))
        x = self.conv12(x, err=self.err[4], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[4], legacy=self.legacy, load_unit=self.load_unit[4], load_wait_w=self.load_wait[4], load_wait_a=self.load_wait[4])
        x = self.tanh(self.bn12(x))
        x = self.conv13(x, err=err_4, forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[4], legacy=self.legacy, load_unit=self.load_unit[4], load_wait_w=self.load_wait[4], load_wait_a=self.load_wait[4])
        x = self.tanh(self.pool(self.bn13(x)))
        x = self.drop5(x)
        
        x = self.fc1(x, err=self.err[5], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[5], legacy=self.legacy, load_unit=self.load_unit[5], load_wait_w=self.load_wait[5], load_wait_a=self.load_wait[5])
        x = x.view(-1,10)
        # BN at the output stabilizes training
        x = self.bnfc1(x)
        return x
    
class CONV_tiny_add_partial(nn.Module):
    '''
    4-layer CNN for SC
    Stream length (err)/load unit/load wait values are specified using a string of 4 integers, one for each layer.
    E.g.: to achieve load unit=2, load wait=2, stream length=32 for all conv layers 128 for the last fc layer, err=[5557], load_unit=[2222], load_wait=[2222]
    '''
    def __init__(self, num_classes=10, c_ins=3, uniform=False, generator='lfsr', sc_compute='1d_bin', legacy=False, relu=False):
        super(CONV_tiny_add_partial, self).__init__()
        self.bn1 = utils_class.BatchNorm2d_fixed(32)
        self.bn2 = utils_class.BatchNorm2d_fixed(32)
        self.bn3 = utils_class.BatchNorm2d_fixed(64)
        self.bn4 = utils_class.BatchNorm1d_fixed(num_classes)
        
        self.conv1 = utils_class.Conv2d_Add_Partial(3, 32, kernel_size=5, padding=2, bias=False)
        self.conv2 = utils_class.Conv2d_Add_Partial(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = utils_class.Conv2d_Add_Partial(32, 64, kernel_size=5, padding=2, bias=False)
        self.fc1   = utils_class.Conv2d_Add_Partial(64, num_classes, kernel_size=4, padding=0, bias=False)
        
        self.tanh = nn.Hardtanh(0,1)
        self.pool = nn.AvgPool2d(2)
        
        self.err = [7,7,7,7]
        self.z_unit = [3,3,4,4]
        self.load_unit = [2,2,2,2]
        self.load_wait_w = [2,2,2,2]
        self.num_classes = num_classes
        self.generator = generator
        self.compute = sc_compute
        self.legacy = legacy
        self.relu = relu
        
        # Scale up weights for low precision and stream length. Other underflow prevents effective training
        if uniform:
            self.conv1.weight_org *= 2
            self.conv2.weight_org *= 2
            self.conv1.weight.data*= 2
            self.conv2.weight.data*= 2
                             
    def forward(self, x_in, target=None):
        x = self.conv1(x_in, err=self.err[0], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[0], legacy=self.legacy, load_unit=self.load_unit[0], load_wait_w=self.load_wait[0], load_wait_a=self.load_wait[0])
        x = self.bn1(x)
        if self.relu:
            x = F.relu(x)
        x = self.pool(x)
        x = self.tanh(x)

        x = self.conv2(x, err=self.err[1], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[1], legacy=self.legacy, load_unit=self.load_unit[1], load_wait_w=self.load_wait[1], load_wait_a=self.load_wait[1])
        x = self.bn2(x)
        if self.relu:
            x = F.relu(x)
        x = self.pool(x)
        x = self.tanh(x)

        x = self.conv3(x, err=self.err[2], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[2], legacy=self.legacy, load_unit=self.load_unit[2], load_wait_w=self.load_wait[2], load_wait_a=self.load_wait[2])
        x = self.bn3(x)
        if self.relu:
            x = F.relu(x)
        x = self.pool(x)
        x = self.tanh(x)

        x = self.fc1(x, err=self.err[3], forward=self.compute, generator=self.generator, z_unit=2**self.z_unit[3], legacy=self.legacy, load_unit=self.load_unit[3], load_wait_w=self.load_wait[2], load_wait_a=self.load_wait[3])
        x = x.view(-1, self.num_classes)
        # BN at the output stabilizes training
        x = self.bn4(x)
        return x
    
class CONV_minimal_add_partial(nn.Module):
    '''
    LeNet CNN for SC
    Stream length of 32 is used for all layers, due the simplicity of MNIST this model is aimed toward
    '''
    def __init__(self, uniform=False, legacy=False, generator='lfsr', sc_compute='1d_bin'):
        super(CONV_minimal_add_partial, self).__init__()
        self.bn1 = utils_class.BatchNorm2d_fixed(6)
        self.bn2 = utils_class.BatchNorm2d_fixed(16)
        
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
            self.conv1.weight_org *= 2
            self.conv2.weight_org *= 2
            self.conv1.weight.data*= 2
            self.conv2.weight.data*= 2
        self.err = 5
        self.legacy = legacy
    def forward(self, x):
        x = self.conv1(x, err=self.err, forward=self.compute, generator=self.generator, legacy=self.legacy)
        x = self.pool(x)
        x = self.bn1(x)
        x = self.tanh(x)
        
        x = self.conv2(x, err=self.err, forward=self.compute, generator=self.generator, legacy=self.legacy)
        x = self.pool(x)
        x = self.bn2(x)
        x = self.tanh(x)
        
        x = x.view(-1, 16*5*5)
        x = self.tanh(self.fc1(x, err=self.err, generator=self.generator))
        x = self.tanh(self.fc2(x, err=self.err, generator=self.generator))
        x = self.fc3(x, err=self.err, generator=self.generator)
        x = self.bn(x)
        return x
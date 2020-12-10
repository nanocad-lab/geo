import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *
import utils_own
import network

import torch.optim as optim
'''
File controlling the overall training and validation procedure
Default parameters train a 4-layer CNN using LFSR generator with 128-bit stream length for all layers. Accumulation is done using OR for x and z dimensions of the filter, while y dimension is accumulated using fixed-point adders.
E.g.: for a 32x5x5 filter, the "32" and the first "5" dimension are accumulated using OR, and the last "5" dimension is accumulated using fixed-point adders, so only 4 integer additions is needed.
'''

parser = argparse.ArgumentParser(description='PyTorch small CNN Training for SC')

parser.add_argument('--save_dir', metavar='SAVE_DIR', default='./training_data_sc/cifar_mid/', type=str, help='save dir. Default translates to ../training_data_sc/cifar_mid')
parser.add_argument('--dataset', metavar='DATASET', default='CIFAR10', type=str, help='dataset to use. Choose between CIFAR10, SVHN, and MNIST')
parser.add_argument('--seed', metavar='SEED', default=0, type=int, help='seed to use for this run')
parser.add_argument('--device', metavar='DEVICE', default=0, type=int, help='the device to use. Device>=0 specifies the GPU ID, device=-1 specifies CPU')
parser.add_argument('--optim', metavar='OPTIM', default='Adam', type=str, help='optimizer to use. Choose between Adam, RMSprop, SGD (with momentum) and Adabound')
parser.add_argument('--lr', metavar='LR', default=2e-3, type=float, help='leaning rate to use')
parser.add_argument('--err', metavar='ERR', default='7777', type=str, help='Err precision per layer. Specifies stream length as 2**Err')
parser.add_argument('--z_unit', metavar='ZUNIT', default='3344', type=str, help='z accumulation unit per layer. Specifies the number of input channles to sum together using OR as 2**z_unit')
parser.add_argument('--load_unit', metavar='LUNIT', default='2222', type=str, help='Number of bits to load each time for progressive loading')
parser.add_argument('--load_wait', metavar='LWAIT', default='2222', type=str, help='Number of cycles to wait between loads for progressive loading')
parser.add_argument('-b','--batch', metavar='BATCH', default=256, type=int, help='Batch size to use')
parser.add_argument('--size', metavar='SIZE', default=0, type=int, help='Size of network to use. Setting 1 uses VGG-16, while default 0 uses a small 4-layer CNN for SVHN and CIFAR-10, and a smaller one for MNIST')
parser.add_argument('--prec', metavar='PRECISION', default='7777', type=str, help='Precision of weight/activation to use. No need to change in general')
parser.add_argument('--val', metavar='VAL', default=0, type=int, help='Evaluate a pretrained model')
parser.add_argument('--uniform', metavar='UNIFORM', default=0, type=int, help='Increase weight magnitude for very short streams (<=32)')
parser.add_argument('--generator', metavar='GEN', default='lfsr', type=str, help='Generator to use. Choose between lfsr, acc (accurate random), and rand (random)')
parser.add_argument('--compute', metavar='COMP', default='1d_bin', type=str, help='SC computation to use. Choose between full_or, 1d_bin, 2d_bin, yz_bin, and z_bin')
parser.add_argument('--legacy', metavar='LEGACY', default=0, type=int, help='Use legacy computation. Disabling it uses accelerated CUDA kernels when possible, and some functions are only available in accelerated versions')
parser.add_argument('--monitor', metavar='MONITOR', default=0, type=int, help='Monitor loss and computation time during training')
parser.add_argument('--relu', metavar='RELU', default=0, type=int, help='Use relu before pooling')

def main():
    global args, best_prec1
    args = parser.parse_args()
    seed = args.seed
    optim_choice = args.optim
    lr = args.lr
    err = args.err
    prec = args.prec
    z_unit = args.z_unit
    generator = args.generator
    compute = args.compute
    load_unit = args.load_unit
    load_wait = args.load_wait
    
    # err, prec, z_unit, load_unit, load_wait are expressed using a string of ints, each representing the value for one
    # layer or a group of layers in the case of VGG-16.
    # Err is used to calculate stream length for each layer. For example, '5557' means stream length is 2**5=32, 2**5=32,
    # 2**5=32, 2**7=128 for each layer
    # Prec is used to specify weight/activation precision. Since stream length directly actual precision, this doesn't 
    # need to be adjusted in general
    errs = []
    precs = []
    z_units = []
    load_units = []
    load_waits = []
    for i in range(len(err)):
        errs.append(int(err[i]))
        precs.append(int(prec[i]))   
        z_units.append(int(z_unit[i]))
        load_units.append(int(load_unit[i]))
        load_waits.append(int(load_wait[i]))
    
    b = args.batch
    
    # Setting seed allows reproducible results
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True

    if args.val == 1:
        val = True
    else:
        val = False
    if args.legacy == 1:
        legacy = True
    else:
        legacy = False
    if args.monitor == 1:
        monitor = True
    else:
        monitor = False
    if args.uniform == 1:
        uniform = True
    else:
        uniform = False
    if args.relu == 1:
        relu = True
    else:
        relu = False

    device = args.device
    dataset = args.dataset
    
    if dataset=='CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                                transform=transforms.Compose([transforms.RandomCrop(32, 4),
                                                                              transforms.RandomHorizontalFlip(),
                                                                              transforms.ToTensor()]))     
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms.ToTensor())    
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=b, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=b, shuffle=False, num_workers=2)
        if args.size==0:
            net = network.CONV_tiny_add_partial(uniform=uniform, sc_compute=compute, generator=generator, legacy=legacy, relu=relu)
            layers = 4
        elif args.size==1:
            net = network.VGG16_add_partial(uniform=uniform, sc_compute=compute, generator=generator, legacy=legacy)
            layers = 6
        if layers != len(err):
            print('Mismatch')
            return -1
        
    elif dataset=='SVHN':
        trainset = torchvision.datasets.SVHN(root='../data', split='train', download=True, transform=transforms.Compose([transforms.RandomCrop(32,4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]))
        testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transforms.Compose([transforms.ToTensor()]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=b, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=b, shuffle=False, num_workers=2)
        smallloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2)
        smalltestloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=2)
        if args.size==0:
            net = network.CONV_tiny_add_partial(uniform=uniform, sc_compute=compute, generator=generator, legacy=legacy, relu=relu)
            layers = 4
        elif args.size==1:
            net = network.VGG16_add_partial(uniform=uniform, sc_compute=compute, generator=generator, legacy=legacy)
            layers = 6
        if layers != len(err):
            print('Mismatch')
            return -1
        
    elif dataset=='MNIST':
        trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                                download=True, transform = transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root='../data', train=False,
                                               download=True, transform = transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                                  shuffle=True, num_workers=2)
        smallloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                                  shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                                 shuffle=False, num_workers=2)
        smalltestloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                                 shuffle=False, num_workers=2)
        net = network.CONV_minimal_add_partial(uniform=uniform, sc_compute=compute, generator=generator, legacy=legacy)
        errs = 4

    # Default save directory is one level up in the hierarchy
    save_dir = os.path.join('../', args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = os.path.join(save_dir, 'CONV_small')

    # Prevents overwriting logging files if already exists
    if not val:
        setup_logging(save_file + '_log.txt')
        logging.info("saving to %s", save_file)
    
    result_dic = save_file + '_result.pt'
    
    save_file += '.pt'
    
    torch.cuda.empty_cache()
    
    # Set model paramters
    net.prec = precs
    net.err = errs
    net.z_unit = z_units
    net.load_unit = load_units
    net.load_wait = load_waits
    if device<0:
        pass
    else:
        net.cuda(device)
        
    if val:
        saved_state_dict = torch.load(save_file, map_location="cpu")
        net.load_state_dict(saved_state_dict, strict=False)
        net.eval()
    
    criterion = nn.CrossEntropyLoss()

    # model specifies the parameters to train on. Since all parameters are trained here, there is no difference between
    # model and net
    model = net

    # device >=0 specifies a GPU
    # This parameter copy is mainly for training with half precision. This functionality is disabled here due to the 
    # limited scaling and instability of half precision training.
    if device>=0:
        param_copy = [param.clone().cuda(device).float().detach() for param in model.parameters()]
    else:
        param_copy = [param.clone().float().detach() for param in model.parameters()]
            
    for param in param_copy:
        param.requires_grad = True
        pass
    
    # Switches between different optimizers
    if optim_choice=='Adam':
        optimizer = optim.Adam(param_copy, lr=lr, weight_decay=0)
    elif optim_choice=='RMSprop':
        optimizer = optim.RMSprop(param_copy, lr=lr, weight_decay=0)
    elif optim_choice=='SGD':
        optimizer = optim.SGD(param_copy, lr=lr, weight_decay=0, momentum=0.01)
    if optim_choice=='Adabound':
        optimizer = adabound.AdaBound(param_copy, lr=lr, final_lr=0.1)
    
    # Learning rate scheduler that anneals learning rate if loss fails decrease for some epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.3, patience=150, verbose=True, threshold=0.05)
    
    best_prec1 = 0
    val_prec_sc = 0
    val_prec1 = 0
    
    if val:
        # Validate accuracy without training
        val_loss, val_prec1, val_prec5 = utils_own.validate(
            testloader, net, criterion, 0, verbal=True, monitor=monitor)
        return 0
    else:
        # Prevents overwriting existing save files
        if os.path.exists(save_file):
            print("Save file already exists. Delete manually if want to overwrite")
            return 0
    

    for epoch in range(0, 1000):  # loop over the dataset multiple times
        # train for one epoch
        train_loss, train_prec1, train_prec5 = utils_own.train(
            trainloader, net, criterion, epoch, optimizer, param_copy=param_copy, modules=model, monitor=monitor)
        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = utils_own.validate(
            testloader, net, criterion, epoch, verbal=True, monitor=monitor)
        # net.add_or = add_or
        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        if is_best:
            torch.save(net.state_dict(), save_file)
            logging.info('\n Epoch: {0}\t'
                         'Training Prec {train_prec1:.3f} \t'
                         'Binary Prec {val_prec1:.3f} \t'
                         'SC Prec {val_prec_sc:.3f} \t'
                         .format(epoch+1, train_prec1=train_prec1, val_prec1=val_prec1, val_prec_sc=val_prec_sc))
        scheduler.step(val_loss)
    
    logging.info('\nTraining finished!')
    return 0
        
if __name__ == '__main__':
    main()

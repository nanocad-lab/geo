import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from utils import *
import utils_own
import network

import torch.optim as optim

import copy
import time
#import adabound

parser = argparse.ArgumentParser(description='PyTorch small CNN Training for SC Darpa')

parser.add_argument('--save_dir', metavar='SAVE_DIR', default='./training_data_sc/cifar_mid/', type=str, help='save dir')
parser.add_argument('--dataset', metavar='DATASET', default='CIFAR10', type=str, help='dataset to use')
parser.add_argument('--seed', metavar='SEED', default=0, type=int, help='seed to used for this run')
parser.add_argument('--max_pool', metavar='MAX_POOL', default=0, type=int, help='use maxpool. Use avgpool if 0')
parser.add_argument('--device', metavar='DEVICE', default=0, type=int, help='the device to use')
parser.add_argument('--optim', metavar='OPTIM', default='Adam', type=str, help='optimizer to use')
parser.add_argument('--lr', metavar='LR', default=2e-3, type=float, help='leaning rate to use')
parser.add_argument('--err', metavar='ERR', default='7777', type=str, help='Err precision per layer')
parser.add_argument('--z_unit', metavar='ZUNIT', default='3344', type=str, help='z accumulation unit per layer')
parser.add_argument('--load_unit', metavar='LUNIT', default='2222', type=str, help='Number of bits to load each time')
parser.add_argument('--load_wait_w', metavar='LWAITW', default='1244', type=str, help='Number of cycles to wait between loads (weight)')
parser.add_argument('--load_wait_a', metavar='LWAITA', default='5555', type=str, help='Number of cycles to wait between loads (activation)')
parser.add_argument('-b','--batch', metavar='BATCH', default=256, type=int, help='Batch size to use')
parser.add_argument('--size', metavar='SIZE', default=0, type=int, help='Size of network to use')
parser.add_argument('-e','--epoch', metavar='EPOCH', default=1000, type=int, help='Number of epochs to train')
parser.add_argument('--prec', metavar='PRECISION', default='7777', type=str, help='Precision of weight/activation to use')
parser.add_argument('--val', metavar='VAL', default=0, type=int, help='Evaluate a pretrained model')
parser.add_argument('--mult_pass', metavar='MULTI_PASS', default=0, type=int, help='Whether or not to train for multi-pass')
parser.add_argument('--uniform', metavar='UNIFORM', default=0, type=int, help='Uniform initialization')
parser.add_argument('--generator', metavar='GEN', default='lfsr', type=str, help='Generator to use')
parser.add_argument('--compute', metavar='COMP', default='1d_bin', type=str, help='SC computation to use')
parser.add_argument('--legacy', metavar='LEGACY', default=0, type=int, help='Use legacy computation')
parser.add_argument('--monitor', metavar='MONITOR', default=0, type=int, help='Monitor computation')
parser.add_argument('--relu', metavar='RELU', default=0, type=int, help='Use relu before pooling')

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

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
    load_wait_w = args.load_wait_w
    load_wait_a = args.load_wait_a
    
    errs = []
    precs = []
    z_units = []
    load_units = []
    load_wait_ws = []
    load_wait_as = []
    for i in range(len(err)):
        errs.append(int(err[i]))
        precs.append(int(prec[i]))   
        z_units.append(int(z_unit[i]))
        load_units.append(int(load_unit[i]))
        load_wait_ws.append(int(load_wait_w[i]))
        load_wait_as.append(int(load_wait_a[i]))
    
    b = args.batch
    e = args.epoch
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True

    if args.max_pool == 1:
        max_pool = True
    else:
        max_pool = False
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
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])
    
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
#             net = network.ResNet18()
        if args.size==1:
            net = network.VGG16_add_partial(uniform=uniform, sc_compute=compute, generator=generator, legacy=legacy, half_pool=half_pool)
            layers = 6
        if layers != len(err):
            print(layers, err, len(err))
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
        if args.size==1:
            net = network.VGG16_add_partial(uniform=uniform, sc_compute=compute, generator=generator, legacy=legacy, half_pool=half_pool)
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

    save_dir = os.path.join('../', args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = os.path.join(save_dir, 'CONV_small')

    if not val:
        setup_logging(save_file + '_log.txt')
        logging.info("saving to %s", save_file)
#         lr /= 100
    
    result_dic = save_file + '_result.pt'
    
    save_file += '.pt'
    
    torch.cuda.empty_cache()
    
    net.prec = precs
    net.err = errs
    net.z_unit = z_units
    net.load_unit = load_units
    net.load_wait_w = load_wait_ws
    net.load_wait_a = load_wait_as
    if device<0:
        pass
    else:
        net.cuda(device)
        
    if val:
        saved_state_dict = torch.load(save_file, map_location="cpu")
        net.load_state_dict(saved_state_dict, strict=False)
        net.eval()
    
    criterion = nn.CrossEntropyLoss()

    model = net

    if device>=0:
        param_copy = [param.clone().cuda(device).float().detach() for param in model.parameters()]
    else:
        param_copy = [param.clone().float().detach() for param in model.parameters()]
            
    for param in param_copy:
        param.requires_grad = True
        
    if optim_choice=='Adam':
        optimizer = optim.Adam(param_copy, lr=lr, weight_decay=0)
    elif optim_choice=='RMSprop':
        optimizer = optim.RMSprop(param_copy, lr=lr, weight_decay=0)
    elif optim_choice=='SGD':
        optimizer = optim.SGD(param_copy, lr=lr, weight_decay=0, momentum=0.01)
    if optim_choice=='Adabound':
        optimizer = adabound.AdaBound(param_copy, lr=lr, final_lr=0.1)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.3, patience=150, verbose=True, threshold=0.05)
    
    best_prec1 = 0
    val_prec_sc = 0
    val_prec1 = 0
    
    if val:
        val_loss, val_prec1, val_prec5 = utils_own.validate(
            testloader, net, criterion, 0, verbal=True, monitor=monitor)
        return 0
    else:
        if os.path.exists(save_file):
            print("Save file already exists. Delete manually if want to overwrite")
            return 0
    

    for epoch in range(0, 1000):  # loop over the dataset multiple times
        # train for one epoch
        train_loss, train_prec1, train_prec5 = utils_own.train(
            trainloader, net, criterion, epoch, optimizer, param_copy=param_copy, modules=model, monitor=monitor)
        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = utils_own.validate(
            testloader, net, criterion, epoch, verbal=True, target=use_target, monitor=monitor)
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

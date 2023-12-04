#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import sys

from train import *
from tools.funcs import *
from tools.TES import TES
from AugNet import UNet
from steganalyzer.CovNet import CovNet
from steganalyzer.YedNet import YedNet
from steganalyzer.SRNet import SRNet
from steganalyzer.LWENet import lwenet
from MyDataloader import MyDataset



def main(args):
    device = torch.device("cuda")

    kwargs = {'num_workers': 4, 'pin_memory': True}

    train_transform = transforms.Compose([
        AugData(),   # AugData() is the D4 augmentation (Baseline)
        ToTensor()
    ])

    eval_transform = transforms.Compose([
        ToTensor()
    ])

    PARAMS_NAME = 'model_params.pt'
    LOG_NAME = 'model_log'
    if not os.path.exists(args.ckt):
        os.mkdir(args.ckt)
    PARAMS_PATH = os.path.join(args.ckt, PARAMS_NAME)   # Path to save models
    LOG_PATH = os.path.join(args.ckt, LOG_NAME)         # Path to save logs

    setLogger(LOG_PATH, mode='w')

    train_dataset = MyDataset(args.cover_dir, args.stego_dir, 0, train_transform)
    valid_dataset = MyDataset(args.cover_dir, args.stego_dir, 1, eval_transform)
    test_dataset = MyDataset(args.cover_dir, args.stego_dir, 2, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size = args.batchsize, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size = args.batchsize, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size = args.batchsize, shuffle=False, **kwargs)

    # the steganalyzer and class-invariant module
    if args.steganalyzer=='CovNet':
        model = CovNet().to(device)  # the steganalyzer
        Teach = CovNet().cuda()     # the class-invariant module (a pretrained steganalyzer)          
    elif args.steganalyzer=='YedNet':
        model = YedNet().to(device)   # the steganalyzer
        Teach = YedNet().to(device)  # the class-invariant module (a pretrained steganalyzer)             
    elif args.steganalyzer=='SRNet':
        model = SRNet().to(device)     # the steganalyzer
        Teach = SRNet().to(device)    # the class-invariant module (a pretrained steganalyzer)   
    elif args.steganalyzer=='LWENet':
        model = lwenet().to(device)    # the steganalyzer
        Teach = lwenet().to(device)    # the class-invariant module (a pretrained steganalyzer)            
    else:
        sys.exit("steganalyzer error (choose CovNet or YedNet or SRNet or LWENet)")

    model = nn.DataParallel(model)
    model.apply(initWeights)

    params = model.parameters()
    params_wd, params_rest = [], []
    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)
    param_groups = [{'params': params_wd, 'weight_decay':5e-4},
                    {'params': params_rest}]

    optimizer = optim.SGD(param_groups, lr=args.lr, momentum=0.9)


    Teach = nn.DataParallel(Teach) # the class-invariant module (a pretrained steganalyzer)  
    all_state = torch.load(args.teach)
    original_state = all_state['original_state']
    Teach.load_state_dict(original_state) 

    # the augmentation network
    AugNet = UNet().cuda()

    # the differentiable sampler
    tes = TES().cuda()

    optimizer_Aug = optim.Adam(AugNet.parameters(), 0.0001)

    # Loading pre-models for the steganalyzer
    if args.load != 'none':         
        all_state = torch.load(args.load)
        original_state = all_state['original_state']
        model.load_state_dict(original_state)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 140, 180], gamma=0.1)
    scheduler_Aug = optim.lr_scheduler.MultiStepLR(optimizer_Aug, milestones=[80, 140, 180], gamma=0.1)
    best_acc = 0.0


    for epoch in range(1, args.epoch + 1):
        scheduler.step()
        scheduler_Aug.step()

        train(model, device, train_loader, optimizer, epoch,AugNet,tes,Teach,optimizer_Aug,args.A,args.num)

        if epoch % EVAL_PRINT_FREQUENCY == 0:
            best_acc = evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH,AugNet)

    logging.info('\nTest set accuracy: \n')

    # Load best network parmater to test
    all_state = torch.load(PARAMS_PATH)
    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)
    evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH,AugNet)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cover_dir', dest='cover_dir', type=str, default='/home/zhangjiansong/dataset/Spatial/cover', help='the path of cover images'
    )
    parser.add_argument(
        '--stego_dir', dest='stego_dir', type=str, default='/home/zhangjiansong/dataset/Spatial/S_UNIWARD_0.4_cover', help='the path of stego images'
    )
    parser.add_argument(
        '--teach', dest='teach', type=str, default='model_params.pt', help='the class-invariant module (a pretrained steganalzer)'
    )
    parser.add_argument(
        '--load', dest='load', type=str, default='model_params.pt', help='Loading pre-models for the steganalyzer'
    )
    parser.add_argument(
        '--lr', dest='lr', type=float, default=0.01, help='learning rate'
    )
    parser.add_argument(
        '--A', dest='A', type=float, default=255, help='the amplitude of the noises'
    )
    parser.add_argument(
        '--num', dest='num', type=float, default=400, help='the number of the noises'
    )
    parser.add_argument(
        '--epoch', dest='epoch', type=float, default=200
    )
    parser.add_argument(
        '--batchsize', dest='batchsize', type=float, default=32
    )
    parser.add_argument(
        '--ckt', dest='ckt', type=str, default='./test/', help='Path to save models and logs'
    )
    parser.add_argument(
        '--steganalyzer', dest='steganalyzer', type=str, default='CovNet', help='CovNet or YedNet or SRNet or LWENet'
    )
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    main(args)
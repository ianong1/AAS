#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import time
import torch
import torch.nn as nn
from tools.funcs import *

TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1

def train(model, device, train_loader, optimizer, epoch,AugNet,tes,Teach,optimizer_Aug,Amp,num):
  batch_time = AverageMeter() 
  data_time = AverageMeter()
  losses = AverageMeter()
  model.train()
  end = time.time()
  for i, sample in enumerate(train_loader):
    data_time.update(time.time() - end) 
    data, label = sample['data'], sample['label']
    shape = list(data.size())
    data = data.reshape(shape[0] * shape[1], *shape[2:])
    label = label.reshape(-1)
    data, label = data.to(device), label.to(device)

    ## adversarial training

    # update the augmentation network
    optimizer_Aug.zero_grad()
    P = AugNet(data)
    noise = tes(P*0.5, P*0.5)  #sampling
    Augdata = data + Amp*noise
    output_target = model(Augdata)  
    output_Teach = Teach(Augdata)  
    criterion = nn.CrossEntropyLoss()
    lossnum = (torch.sum(abs(noise),dim=[1,2,3]).mean()-num)**2
    loss = criterion(output_target, 1-label) + criterion(output_Teach, label) + lossnum
    loss.backward()       
    optimizer_Aug.step()

    # update the steganalyzer
    optimizer.zero_grad()
    end = time.time()
    P = AugNet(data)
    noise = tes(P*0.5, P*0.5)
    Augdata = data + Amp*noise
    output1 = model(Augdata.detach()) 
    criterion = nn.CrossEntropyLoss()
    loss1 = criterion(output1, label)
    output2 = model(data)  
    loss2 = criterion(output2, label)
    loss = loss1 + loss2
    losses.update(loss.item(), data.size(0))
    loss.backward()       
    optimizer.step()

    batch_time.update(time.time() - end) 
    end = time.time()

    if i % TRAIN_PRINT_FREQUENCY == 0:
      logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))



def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH,AugNet):
  model.eval()

  correct = 0.0

  with torch.no_grad():
    for sample in eval_loader:
      data, label = sample['data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)

      data, label = data.to(device), label.to(device)

      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(label.view_as(pred)).sum().item()

  accuracy = correct / (len(eval_loader.dataset) * 2)

  if accuracy > best_acc and epoch > 140:
    best_acc = accuracy
    all_state = {
      'original_state': model.state_dict(),
      'AugNet_state': AugNet.state_dict(),
      'optimizer_state': optimizer.state_dict(),
      'epoch': epoch
    }
    torch.save(all_state, PARAMS_PATH)
  
  logging.info('-' * 8)
  logging.info('Eval accuracy: {:.4f}'.format(accuracy))
  logging.info('Best accuracy:{:.4f}'.format(best_acc))   
  logging.info('-' * 8)
  return best_acc


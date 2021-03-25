import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util import *
from cifar_models import *


## Parser setting
parser = argparse.ArgumentParser(description='CIFAR TRAIN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=0.001, type=float, dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, dest='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, dest='weight_decay')

parser.add_argument('--batch_size', default=128, type=int, dest='batch_size')
parser.add_argument('--num_epoch', default=300, type=int, dest='num_epoch')

parser.add_argument('--num_workers', default=8, type=int, dest='num_workers')

parser.add_argument('--data_dir', default='./data', type=str, dest='data_dir')
parser.add_argument('--ckpt_dir', default='./checkpoint', type=str, dest='ckpt_dir')

parser.add_argument('--cifar_type', default=10, type=int, dest='cifar_type')
parser.add_argument('--network', default='VGGNet', type=str, dest='network')

parser.add_argument('--mode', default='train', type=str, dest='mode')
parser.add_argument('--train_continue', default='off', type=str, dest='train_continue')

parser.add_argument('--vgg_type', default='A', type=str, dest='vgg_type')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    global best_acc
    
    print('----------------Preparing CIFAR dataset----------------')
    
    ## TRAIN MODE
    if args.mode == 'train':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        if args.cifar_type == 10:
            dataloader = datasets.CIFAR10
            num_classes = 10
        elif args.cifar_type == 100:
            dataloader = datasets.CIFAR100
            num_classes = 100
        
        train_dataset = dataloader(root=args.data_dir, train=True, download=True, transform=transform_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        if args.network.startswith('VGGNet'):
            net = VGGNet(vgg_type=args.vgg_type, num_classes=num_classes)  

        net = torch.nn.DataParallel(net).to(device)
        print('----------------Total Parameters: %.2fM----------------' % (sum(p.numel() for p in net.parameters()) / 1000000.0))

        fn_loss = nn.CrossEntropyLoss()

        optim = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        st_epoch = 0

        if args.train_continue == 'on':
            net, optim, st_epoch = load(ckpt_dir=args.ckpt_dir, net=net, optim=optim)

        for epoch in range(st_epoch + 1, args.num_epoch + 1):
            train_loss, train_acc = one_epoch_train(train_loader, net, fn_loss, optim, epoch)
            print('TRAIN: EPOCH %04d / %04d | LOSS %.4f | ACC %.4f' %
                    (epoch, args.num_epoch, train_loss, train_acc))

            if epoch % 20 == 0:
                save(ckpt_dir=args.ckpt_dir, net=net, optim=optim, epoch=epoch)

    ## TEST MODE
    elif args.mode == 'test':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.cifar_type == 10:
            dataloader = datasets.CIFAR10
            num_classes = 10
        elif args.cifar_type == 100:
            dataloader = datasets.CIFAR100
            num_classes = 100

        test_dataset = dataloader(root=args.data_dir, train=False, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        if args.network.startswith('VGGNet'):
            net = VGGNet(vgg_type=args.vgg_type, num_classes=num_classes)            

        net = torch.nn.DataParallel(net).to(device)
        print('----------------Total Parameters: %.2fM----------------' % (sum(p.numel() for p in net.parameters()) / 1000000.0))

        fn_loss = nn.CrossEntropyLoss()

        optim = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        net, optim, st_epoch = load(ckpt_dir=args.ckpt_dir, net=net, optim=optim)

        for epoch in range(st_epoch + 1, args.num_epoch + 1):
            test_loss, test_acc = one_epoch_test(test_loader, net, fn_loss, epoch)
            print('TEST: EPOCH %04d / %04d | LOSS %.4f | ACC %.4f' %
                    (epoch, args.num_epoch, test_loss, test_acc))

            best_acc = test_acc if test_acc > best_acc else best_acc

        print('%s BEST ACCURACY: %.4f' % (args.network, best_acc))


def one_epoch_train(train_loader, net, fn_loss, optim, epoch):
    net.train()

    batch_time = Average()
    data_time = Average()
    #loss_arr = Average()
    #top1 = Average()
    #top5 = Average()

    end = time.time()

    correct = 0
    loss_arr = 0

    for batch, (inputs, labels) in enumerate(train_loader, 1):
        # print('state" %04d' % batch)

        data_time.update(time.time() - end)

        inputs = torch.autograd.Variable(inputs.to(device))
        labels = torch.autograd.Variable(labels.to(device))

        outputs = net(inputs)
        loss = fn_loss(outputs, labels)

        #pred1, pred5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        #loss_arr.update(loss.data[0], inputs.size(0))
        #top1.update(pred1[0], inputs.size(0))
        #top5.update(pred5[0], inputs.size(0))
        loss_arr += loss.item()

        _, predict = outputs.max(1)
        correct += predict.eq(labels).sum().item() 

        optim.zero_grad()
        loss.backward()
        optim.step()

        batch_time.update(time.time() - end)
        end = time.time()

    acc = correct / batch
    loss = loss_arr / batch

    #return (loss_arr.avg, top1.avg)
    return loss, acc

def one_epoch_test(test_loader, net, fn_loss, epoch):
    global best_acc
    
    net.eval()

    batch_time = Average()
    data_time = Average()
    #loss_arr = Average()
    #top1 = Average()
    #top5 = Average
    
    end = time.time()

    correct = 0
    loss_arr = 0

    for batch, (inputs, labels) in enumerate(test_loader, 1):
        data_time.update(time.time() - end)

        inputs = torch.autograd.Variable(inputs.to(device))
        labels - torch.autograd.Variable(labels.to(device))

        outputs = net(inputs)
        loss = fn_loss(outputs, labels)

        #pred1, pred5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        #loss_arr.update(loss.data[0], inputs.size(0))
        #top1.update(pred1[0], inputs.size(0))
        #top5.update(pred5[0], inputs.size(0))
        loss_arr += loss.item()

        _, predict = outputs.max(1)
        correct += predict.eq(labels).sum().item() 

        batch_time.update(time.time() - end)
        end = time.time()
    
    acc = correct / batch
    loss = loss_arr / batch

    #return (loss_arr.avg, top1.avg)
    return loss, acc




if __name__ == '__main__':
    main()
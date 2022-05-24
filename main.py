"""
Training script for Hyperspectral Image Classification
Copyright (c) Zhiqiang Gong, 2021
"""
from __future__ import print_function
from concurrent.futures.process import _threads_wakeups

import os
import time
import tqdm
import argparse
import shutil
import random
import scipy.io as sio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from pytorch_metric_learning import miners, losses

from torch.utils.data import DataLoader, random_split

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from datasets import HyperDataset, data_info
from net.models import D3_Net_Memory, D3_CNN
from net.SSFTTnet import SSFTTnet

import net.TPPI.models as models

# Parse arguments
parser = argparse.ArgumentParser(description="PyTorch Hyperspectral Training")

# Datasets
parser.add_argument(
    "-r", "--root", default="hyperspectral_dataset/", type=str
)
parser.add_argument("-d", "--dataset", default="pavia", type=str)
parser.add_argument(
    "-m", "--method", default="3DCNN", type=str
)
parser.add_argument("--neighbor", default=5, type=int, help="neighbors of pixel")
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
# Optimization options
parser.add_argument(
    "--epochs", default=500, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--train-batch",
    default=4,
    type=int,
    metavar="N",
    help="train batchsize (default: 16)",
)
parser.add_argument(
    "--test-batch",
    default=4,
    type=int,
    metavar="N",
    help="test batchsize (default: 200)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.0001,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "--drop",
    "--dropout",
    default=0,
    type=float,
    metavar="Dropout",
    help="Dropout ratio",
)
parser.add_argument(
    "--schedule",
    type=int,
    nargs="+",
    default=[150, 225],
    help="Decrease learning rate at these epochs.",
)
parser.add_argument(
    "--gamma", type=float, default=0.1, help="LR is multiplied by gamma on schedule."
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")

parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-5,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
# Checkpoints

parser.add_argument(
    "-c",
    "--checkpoint",
    default="checkpoint",
    type=str,
    metavar="PATH",
    help="path to save checkpoint (default: checkpoint)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)

parser.add_argument(
    "--test_path",
    default="checkpoint/model_best.pth.tar",
    #default="pavia/7_7/1/model_best.pth.tar",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
# Architecture

# Miscs
parser.add_argument("--manualSeed", type=int, help="manual seed")
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
# Device options
parser.add_argument(
    "--gpu", default="0", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)

# data augmentation
parser.add_argument('--flip_augmentation', action='store_true',
                    help="Random flips (if patch_size > 1)")
parser.add_argument('--radiation_augmentation', action='store_true',
                    help="Random radiation noise (illumination)")
parser.add_argument('--mixture_augmentation', action='store_true',
                    help="Random mixes between spectra")

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# print(state)
# Use CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

hyperparams = vars(args)    # 以字典类型返回解析参数值，存入hyperparams

# Number of classes


def main():
    global best_acc

    print('Starting......')
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    root = args.root
    data = args.dataset
    neighbor = args.neighbor

    # create model00
    input_dim, num_classes = data_info(args.dataset)
    
    IGNORED_LABELS = [0]

    ind=False
    if args.method=='SSFTTnet':
        ind = True

    trainval_dataset = HyperDataset(root, data, num=1, train=True, shuffle=True, neighbor=neighbor, ind=ind)
    test_dataset = HyperDataset(root, data, num=1, train=False, neighbor=neighbor, ind=ind)

    train_length, val_length = int(len(test_dataset) * 0.9), len(
        test_dataset
    ) - int(len(test_dataset) * 0.9)
    train_dataset, val_dataset = random_split(
        test_dataset, [train_length, val_length]
    )

    print(
        f"Prepared dataset, train:{int(len(train_dataset))},\
            val:{int(len(val_dataset))}, test:{len(test_dataset)}"
    )

    train_loader = DataLoader(
        trainval_dataset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        test_dataset,
        #val_dataset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.method=='3DCNN':
    	model = D3_CNN(num_classes, in_channels=input_dim,patch_size=args.neighbor)
    elif args.method=='SSFTTnet':
    	model = SSFTTnet(num_classes=num_classes,BATCH_SIZE_TRAIN=args.train_batch)
    elif args.method=='NoiPhy':
    	model = D3_Net_Memory(num_classes, in_channels=input_dim,patch_size=args.neighbor)
    elif args.method=='PResNet':
    	model = models.pResNet(args.dataset)
    elif args.method=='HybridSN':
    	model = models.HybridSN(args.dataset)
    else:
    	raise ValueError("The method is not supported!")
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    cir_loss = losses.CircleLoss()
    miner = miners.MultiSimilarityMiner()
    criterion_metric = losses.TripletMarginLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        #momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # Resume
    title = data
    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint["epoch"]
        print(checkpoint["state_dict"])
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger = Logger(
            os.path.join(args.checkpoint, "log.txt"), title=title, resume=True
        )
    else:
        logger = Logger(os.path.join(args.checkpoint, "log.txt"), title=title)
        logger.set_names(
            ["Learning Rate", "Train Loss", "Valid Loss", "Train Acc.", "Valid Acc."]
        )

    if args.evaluate:
        print("\nEvaluation only")

        print(args.test_path)
        assert os.path.isfile(args.test_path), "Error: no checkpoint directory found!"
        args.checkpoint = os.path.dirname(args.test_path)
        checkpoint = torch.load(args.test_path)
        best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])

        test_loss, test_acc = test(test_loader, model, criterion, start_epoch, use_cuda)
        print(" Test Loss:  %.8f, Test Acc:  %.2f" % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(
            train_loader, model, criterion, criterion_metric, miner, cir_loss, optimizer, epoch, use_cuda
        )
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state["lr"], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "acc": test_acc,
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            checkpoint=args.checkpoint,
        )

    logger.close()
    #logger.plot()
    savefig(os.path.join(args.checkpoint, "log.eps"))

    print("Best acc:")
    print(best_acc)


def train(train_loader, model, criterion, criterion_metric, miner, cir_loss, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    tq_loader = tqdm.tqdm(train_loader)

    for _, (inputs, targets) in enumerate(tq_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = (
                inputs.type(torch.FloatTensor).cuda(),
                targets.type(torch.LongTensor).cuda(),
            )

        # compute output
        outputs = model(inputs)

        #clc_metric = cir_loss(embeddings, targets)
        #print('loss_1:', criterion(outputs, targets))
        #print('loss_2', clc_metric)
        loss = criterion(outputs, targets) #+ 1e-1 * clc_metric 

        # measure accuracy and record loss
        prec1, _, _, _ = accuracy(outputs.data, targets.data)

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        tq_loader.set_description("Epoch: [%d | %d]" % (epoch + 1, args.epochs))
        tq_loader.set_postfix(loss=loss.item())
    return (losses.avg, top1.avg)


def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    tq_loader = tqdm.tqdm(val_loader)

    pred_labels = None
    ground_labels = None

    for _, (inputs, targets) in enumerate(tq_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = (
                inputs.type(torch.FloatTensor).cuda(),
                targets.type(torch.LongTensor).cuda(),
            )

        # print(inputs)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, _, pred, ground = accuracy(outputs.data, targets.data)
        #print(outputs.data)
        #print(targets.data)

        if pred_labels == None:
            pred_labels = pred
            ground_labels = ground
        else:
            pred_labels = torch.cat((pred_labels, pred), 1)
            ground_labels = torch.cat((ground_labels, ground), 1)

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        tq_loader.set_description("Val: [%d | %d]" % (epoch + 1, args.epochs))
        tq_loader.set_postfix(acc=top1.avg.item())

    #print(pred_labels.cpu().numpy())
    np.savetxt('results/predict.txt', pred_labels.cpu().numpy())
    np.savetxt('results/ground.txt', ground_labels.cpu().numpy())
    sio.savemat('results/predict.mat', {'predict': pred_labels.cpu().numpy(), 'ground': ground_labels.cpu().numpy()})
    return (losses.avg, top1.avg)


def save_checkpoint(
    state, is_best, checkpoint="checkpoint", filename="checkpoint.pth.tar"
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state["lr"] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group["lr"] = state["lr"]


if __name__ == "__main__":
    main()

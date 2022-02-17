from __future__ import print_function

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from engine import train, meta_test
from losses import SIMCLRLoss
from resnet import resnet12
from tiered_imnet import TieredImageNet, MetaTieredImageNet


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    # dataset
    parser.add_argument('--model', type=str, default='resnet12')
    parser.add_argument('--dataset', type=str, default='tieredImageNet')
    parser.add_argument('--data-root', type=str, default='/home/max/datasets/Imagenet/train',
                        help='path to data root')
    parser.add_argument('--input-size', default=84, type=int, help='images input size')
    parser.add_argument('--batch-size', type=int, default=64, help='batch_size')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    parser.add_argument('--output_dir', type=str, default='/home/max/Documents/gpt3-gen-fs')

    parser.add_argument("--word-embeddings", default="./data/gpt3_txt_bert_uncased_L-8_H-512_A-8_embed", type=str)

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')


    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')

    # augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # meta setting
    parser.add_argument('--num_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--num_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--num_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--num_queries', type=int, default=15, metavar='N',  # changed from 15 to 5
                        help='Number of query in test')
    parser.add_argument('--augmentations', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--meta_test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin-mem', default=True)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--resume", default="", type=str, help="resume from a checkpoint")
    parser.add_argument("--meta_test", default=False, type=bool)

    return parser.parse_args()


def main(args):
    utils.init_distributed_mode(args)
    args.device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    train_ds = TieredImageNet(args=args, partition='train', word_embeddings=args.word_embeddings)
    meta_test_ds = MetaTieredImageNet(args=args, partition='test', word_embeddings=args.word_embeddings,
                                      num_ways=args.num_ways, num_shots=args.num_shots,
                                      num_queries=args.num_queries, augmentations=args.augmentations,
                                      num_test_runs=args.num_test_runs)

    # no longer relevant for base class
    args.nb_classes = 351

    sampler_train = torch.utils.data.DistributedSampler(
        train_ds, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_mem
    )

    meta_loader = torch.utils.data.DataLoader(
        meta_test_ds, sampler=torch.utils.data.SequentialSampler(meta_test_ds),
        batch_size=args.meta_test_batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem
    )

    # model
    model = resnet12()
    model.to(args.device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # optimizer
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     weight_decay=0.0005)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

    # criterion = nn.CrossEntropyLoss()
    criterion = SIMCLRLoss()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    # set cosine annealing scheduler
    if args.cosine:
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min, -1)

    best_acc1 = 0.0
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Number of params:', n_parameters)

    output_dir = Path(args.output_dir)

    # routine: self-supervised pre-training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_stats = train(epoch, train_loader, model, criterion, optimizer, args.device)
        if epoch % 2 == 0 or epoch == args.epochs - 1:
            # zero_shot_stats = zero_shot(model_without_ddp, meta_loader, args)
            zero_shot_stats = meta_test(model_without_ddp, meta_loader, args)

            acc1 = zero_shot_stats['acc1']
            is_best = acc1 > best_acc1
            best_acc1 = max(best_acc1, acc1)

            print(f"Meta Accuracy of the network: {zero_shot_stats['acc1']: .1f}%")
            print(f'Max meta test accuracy: {best_acc1:.2f}%')

            log_stats = {
                **{f'base_train_{k}': v for k, v in train_stats.items()},
                **{f'meta_test_{k}': v for k, v in zero_shot_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters,
                "lr": args.learning_rate
            }

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            print("=> saving checkpoint")
            utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'scaler': scaler.state_dict(),
                'best_acc1': best_acc1,
                'args': args,
            }, is_best, args.output_dir)

        if args.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, args, optimizer)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def adjust_learning_rate(epoch, args, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
    if steps > 0:
        new_lr = args.learning_rate * (args.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


if __name__ == '__main__':
    args = parse_option()
    if args.output_dir:
        args.output_dir = f"{args.output_dir}/{args.dataset}/{args.model}/{args.num_ways}-way/{args.num_shots}-shot" \
                          f"/simclr/linear-meta-eval/gpt3"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

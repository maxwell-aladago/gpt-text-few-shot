import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from models import vit_tiny_patch16_224
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from losses import SIMCLRLoss
import utils
from engine import train, zero_shot
from tiered_imnet import TieredImageNet, MetaTieredImageNet


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--model', type=str, default='vit_tiny_patch16_224')
    parser.add_argument('--batch-size', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument("--word-embeddings", default="./data/gpt3_txt_bert_uncased_L-8_H-512_A-8_embed", type=str)
    # parser.add_argument("rep-size", default=512, type=int, help="the projection_dim", required=False)

    # dataset
    parser.add_argument('--dataset', type=str, default='tieredImageNet')
    parser.add_argument('--data-root', type=str, default='/home/max/datasets/Imagenet/train',
                        help='path to data root')

    parser.add_argument('--output_dir', type=str, default='/home/max/Documents/gpt3-gen-fs')

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

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.005,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                                 "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # device and hardware args
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin-mem', default=True)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

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
    model = vit_tiny_patch16_224(rep_size=512)
    model.to(args.device)

    # TODO: Load model from saved checkpoints
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)

    lr_scheduler, _ = create_scheduler(args, optimizer)

    # TODO: change this loss function [partially done]
    # TODO: think about employing loss scaler
    # criterion = nn.CrossEntropyLoss()
    criterion = SIMCLRLoss()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Number of params:', n_parameters)

    output_dir = Path(args.output_dir)

    # routine: self-supervised pre-training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_acc1 = 0.0
    for epoch in range(args.epochs):
        train_stats = train(epoch, train_loader, model, criterion, optimizer, args.device)
        if epoch % 2 == 0:
            zero_shot_stats = zero_shot(model_without_ddp, meta_loader, args)

            acc1 = zero_shot_stats['acc1']
            is_best = acc1 > best_acc1
            best_acc1 = max(best_acc1, acc1)

            print("=> saving checkpoint")
            utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'scaler': scaler.state_dict(),
                'best_acc1': best_acc1,
                'args': args,
            }, is_best, args.output_dir)

            print(f"Meta Accuracy of the network: {zero_shot_stats['acc1']: .1f}%, {zero_shot_stats['acc5']: .1f}%")
            print(f'Max meta test accuracy: {best_acc1:.2f}%')

            log_stats = {
                **{f'base_train_{k}': v for k, v in train_stats.items()},
                **{f'meta_test_{k}': v for k, v in zero_shot_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters,
                "lr": args.lr
            }

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        lr_scheduler.step(epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = parse_option()
    if args.output_dir:
        args.output_dir = f"{args.output_dir}/{args.dataset}/{args.model}/{args.num_ways}-way/{args.num_shots}-shot"

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

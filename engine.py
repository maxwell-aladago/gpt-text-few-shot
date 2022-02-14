import torch
import utils
from timm.utils import accuracy
import math
import sys
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from models import MLPHead
from losses import MetaSIMCLRLoss


def train(epoch, train_loader, model, criterion, optimizer, device):
    """One epoch training"""
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, word_embed, target in metric_logger.log_every(train_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        word_embed = word_embed.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(samples)
            loss_dict = criterion(output, word_embed)
            loss = loss_dict['loss']

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        # ===================meters=====================

        torch.cuda.synchronize()
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = target.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(loss_dict['acc'].item(), n=batch_size)
        metric_logger.meters['loss'].update(loss.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def meta_test(net, test_loader, args):
    net = net.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Meta_Test:'

    for i, (supp_x, supp_y, query_x, query_y) in enumerate(metric_logger.log_every(test_loader, 100, header)):
        bs, n_s, c, h, w = supp_x.size()

        supp_x = supp_x.view(bs * n_s, c, h, w)
        n_q = query_x.shape[1]
        query_x = query_x.view(bs * n_q, c, h, w)

        supp_x = supp_x.to(args.device, non_blocking=True)
        with torch.cuda.amp.autocast():
            supp_x = net(supp_x, return_base_features=True)
            supp_x = F.normalize(supp_x, dim=-1)

        # move support tensors to cpu so query tensors can have gpu memory
        supp_x = supp_x.detach().cpu()
        query_x = query_x.to(args.device, non_blocking=True)
        with torch.cuda.amp.autocast():
            query_x = net(query_x, return_base_features=True)
            query_x = F.normalize(query_x, dim=-1)

        # empty gpu memory
        query_x = query_x.detach().cpu()

        supp_x = supp_x.numpy()
        query_x = query_x.numpy()

        supp_y = supp_y.reshape(bs * n_s).numpy()
        query_y = query_y.reshape(bs * n_q).numpy()

        clf = LogisticRegression(penalty='l2',
                                     random_state=0,
                                     C=1.0,
                                     solver='lbfgs',
                                     max_iter=1000,
                                     multi_class='multinomial')

        clf.fit(supp_x, supp_y)
        query_y_pred = clf.predict(query_x)
        supp_y_pred = clf.predict(supp_x)

        batch_size = query_y.shape[0]
        acc1 = metrics.accuracy_score(query_y, query_y_pred) * 100
        supp_acc1 = metrics.accuracy_score(supp_y, supp_y_pred) * 100

        metric_logger.meters['acc1'].update(acc1, n=batch_size)
        metric_logger.meters['supp_acc1'].update(supp_acc1, n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} '.format(top1=metric_logger.acc1))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def zero_shot(net, test_loader, args):
    net = net.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Meta_Test:'

    word_embed = test_loader.dataset.we_meta
    word_embed = word_embed.to(args.device)

    # generate projection later
    # generate loss function,
    # generate optimizer
    # write a function to train projection

    with torch.torch.no_grad():
        for i, (supp_x, supp_y, query_x, query_y) in enumerate(metric_logger.log_every(test_loader, 100, header)):
            bs, n_s, c, h, w = supp_x.size()
            # supp_x = supp_x.view(bs * n_s, c, h, w)
            # supp_y = supp_y.reshape(bs * n_s)
            n_q = query_x.shape[1]
            query_x = query_x.view(bs * n_q, c, h, w)
            query_y = query_y.reshape(bs * n_q)

            query_x = query_x.to(args.device, non_blocking=True)
            query_y = query_y.to(args.device, non_blocking=True)

            query_x = net(query_x, return_base_features=False)
            query_x = query_x / query_x.norm(dim=1, keepdim=True)

            logits_per_image = query_x @ word_embed.t()

            acc1, acc5 = accuracy(logits_per_image, query_y, topk=(1, 5))

            # empty gpu memory
            query_x = query_x.detach().cpu()
            query_y = query_y.detach().cpu()

            batch_size = query_y.shape[0]

            metric_logger.meters['acc1'].update(acc1, n=batch_size)
            metric_logger.meters['acc5'].update(acc5, n=batch_size)

        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} '.format(top1=metric_logger.acc1))
        word_embed = word_embed.detach().cpu()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_meta_model(support_inputs, device, out_dim=512, max_iters=200):
    supp_x, supp_text, supp_y = support_inputs[0], support_inputs[1], support_inputs[2]

    in_dim = supp_x.shape[-1]
    model = MLPHead(in_chans=in_dim, rep_size=out_dim)

    model = model.to(device)
    supp_x = supp_x.to(device)
    supp_y = supp_y.to(device)
    supp_text = supp_text.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)
    criterion = MetaSIMCLRLoss()

    acc_ = []
    loss_ = []
    for i in range(max_iters):
        optimizer.zero_grad()
        pred_x = model(supp_x)
        loss_dict = criterion(pred_x, supp_text, supp_y)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()
        acc_.append(loss_dict['acc'].item())
        loss_.append(loss.item())

    return model, torch.tensor(acc_).mean().item(), torch.tensor(loss_).mean().item()


def meta_test_proj(net, test_loader, args):
    net = net.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Meta_Test:'

    for i, (supp_inputs, query_x, query_y) in enumerate(metric_logger.log_every(test_loader, 100, header)):
        supp_x, supp_y = supp_inputs[0], supp_inputs[2]

        bs, n_s, c, h, w = supp_x.size()
        supp_x = supp_x.view(bs * n_s, c, h, w)
        n_q = query_x.shape[1]
        query_x = query_x.view(bs * n_q, c, h, w)
        supp_y = supp_y.reshape(bs * n_s)
        query_y = query_y.reshape(bs * n_q)

        supp_x = supp_x.to(args.device, non_blocking=True)
        with torch.no_grad():
            supp_x = net(supp_x, use_hidden_state=True)
            supp_x = supp_x / supp_x.norm(dim=1, keepdim=True)

        supp_inputs[0] = supp_x
        supp_inputs[2] = supp_y
        model, supp_acc, supp_loss = train_meta_model(supp_inputs, out_dim=args.n_ways, device=args.device0)
        supp_x = supp_x.detach().cpu()

        acc1, acc5 = eval_zero_shot(net, model, test_loader.dataset.we_meta, query_x, query_y, args.device)
        batch_size = query_y.shape[0]
        metric_logger.meters['acc1'].update(acc1, n=batch_size)
        metric_logger.meters['acc5'].update(acc5, n=batch_size)
        metric_logger.meters['supp_loss'].update(supp_loss, n=batch_size)
        metric_logger.meters['supp_acc'].update(supp_acc, n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} '.format(top1=metric_logger.acc1))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_zero_shot(base_model, linear_proj, word_embed, query_x, query_y, device):
    base_model.eval()
    linear_proj.eval()
    word_embed = word_embed.to(device)
    with torch.torch.no_grad():
        bs, n_q, c, h, w = query_x.shape
        query_x = query_x.view(bs * n_q, c, h, w)
        query_y = query_y.reshape(bs * n_q)

        query_x = query_x.to(device, non_blocking=True)
        query_y = query_y.to(device, non_blocking=True)

        query_x = base_model(query_x, return_base_features=False)
        query_x = query_x / query_x.norm(dim=1, keepdim=True)

        query_x = linear_proj(query_x)

        logits_per_image = query_x @ word_embed.t()

        acc1, acc5 = accuracy(logits_per_image, query_y, topk=(1, 5))

        # empty gpu memory
        query_x = query_x.detach().cpu()
        query_y = query_y.detach().cpu()
        word_embed = word_embed.detach().cpu()

        return acc1, acc5

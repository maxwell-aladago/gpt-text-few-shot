import torch
import utils
from timm.utils import accuracy
import math
import sys
from torch import nn


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


# def get_meta_model(in_dim, out_dim):
#     # linear projection
#     model = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
#     nn.init.kaiming_normal_(model.weight.data)
#     nn.init.zeros_(model.bias.data)
#
#     return model
#
#
# def train_meta_model(support_x, support_y, out_dim, max_iters, device):
#     in_dim = support_x.shape[-1]
#     model = get_meta_model(in_dim, out_dim)
#     model = model.to(device)
#     support_x = support_x.to(device)
#     support_y = support_y.to(device)
#
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()
#     total_loss = 0.0
#     for i in range(max_iters):
#         optimizer.zero_grad()
#         pred_x = model(support_x)
#         loss = criterion(pred_x, support_y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#
#     return model, total_loss / max_iters
#
#
# def meta_test_proj(net, test_loader, args):
#     net = net.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Meta_Test:'
#
#     for i, (supp_x, supp_y, query_x, query_y) in enumerate(metric_logger.log_every(test_loader, 100, header)):
#         bs, n_s, c, h, w = supp_x.size()
#         supp_x = supp_x.view(bs * n_s, c, h, w)
#         n_q = query_x.shape[1]
#         query_x = query_x.view(bs * n_q, c, h, w)
#         supp_y = supp_y.reshape(bs * n_s)
#         query_y = query_y.reshape(bs * n_q)
#
#         supp_x = supp_x.to(args.device, non_blocking=True)
#         with torch.no_grad():
#             supp_x = net(supp_x, use_hidden_state=True)
#             supp_x = supp_x / supp_x.norm(dim=1, keepdim=True)
#
#         model, supp_loss = train_meta_model(supp_x, supp_y, out_dim=args.n_ways, device=args.device, max_iters=20)
#         supp_x = supp_x.detach().cpu()
#
#         model.eval()
#         with torch.no_grad():
#             query_x = query_x.to(args.device, non_blocking=True)
#             query_x = net(query_x, use_hidden_state=True)
#             query_x = query_x / query_x.norm(dim=1, keepdim=True)
#             pred_query = model(query_x)
#             pred_query = pred_query.argmax(dim=1).squeeze()
#             query_x = query_x.detach().cpu()
#
#         batch_size = query_y.shape[0]
#         acc1 = metrics.accuracy_score(query_y, pred_query.cpu().numpy()) * 100
#         metric_logger.meters['acc1'].update(acc1, n=batch_size)
#         metric_logger.meters['supp_loss'].update(supp_loss, n=batch_size)
#
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} '.format(top1=metric_logger.acc1))
#
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

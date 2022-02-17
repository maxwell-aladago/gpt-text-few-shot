import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

plt.rcParams["font.size"] = 25


def process_logfile(fpath):
    top_1 = []
    meta_top1 = []
    lr = []
    with open(fpath, 'r') as logf:
        for line in logf:
            line = json.loads(line)
            top_1.append(line['meta_test_acc1'])
            meta_top1.append(line['base_train_acc1'])
            lr.append(line['base_train_lr'])

    return top_1, meta_top1, lr


def meta_log(fpath):
    supp_acc = []
    query_acc = []
    with open(fpath, 'r') as logf:
        for line in logf:
            line = json.loads(line)
            supp_acc.append(line['meta_test_supp_acc1'])
            query_acc.append(line['meta_test_acc1'])

    return supp_acc, query_acc


def bert_small_few_shot():
    plt.rcParams["font.size"] = 25
    base_path = '/home/max/Documents/gpt3-gen-fs/tieredImageNet/' \
                '/vit_tiny_patch16_224/5-way/1-shot/linear-meta-eval'

    methods = [
        "gpt3-text-5e-3",
        "template",
        "integer-label",
        "rand"
    ]

    # colors = ['purple', 'blue', 'orange', 'brown']
    colors = ['green', 'blue', 'orange', 'brown']
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(38, 12))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Base Training Accuracy")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Meta Test Accuracy")

    for i, model in enumerate(methods):
        path_ = f"{base_path}/{model}/log.txt"
        test_top1, train_top1, lr = process_logfile(path_)
        x = np.arange(0, len(test_top1), step=1) * 2
        ax2.plot(x, test_top1, color=colors[i], linestyle='-', lw=2)
        ax1.plot(x, train_top1, color=colors[i], linestyle='-', lw=2)
        ax3.plot(x, lr, color=colors[i], linestyle='-', lw=2)

        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    labels = ["GPT3-Features", "Templates", "Integer Labels", "Random Vectors"]

    ax1.legend(labels)
    # fig.legend(handles=ax1.lines,
    #            labels=labels,
    #            loc="lower center",
    #            fancybox=False,
    #            ncol=4,
    #            bbox_to_anchor=(0.01, -0.02, 0.9, 0.02),
    #            frameon=False,
    #            borderaxespad=0.,
    #            mode='expand')

    plt.savefig(f"plots/vit-tiny-bert-medium-linear-meta-train.pdf", bbox_inches='tight')


if __name__ == '__main__':
    bert_small_few_shot()

import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

plt.rcParams["font.size"] = 25


def process_logfile(fpath):
    top_1 = []
    meta_top1 = []
    with open(fpath, 'r') as logf:
        for line in logf:
            line = json.loads(line)
            top_1.append(line['meta_test_acc1'])
            meta_top1.append(line['base_train_acc1'])

    return top_1, meta_top1


def meta_log(fpath):
    supp_acc = []
    query_acc = []
    with open(fpath, 'r') as logf:
        for line in logf:
            line = json.loads(line)
            supp_acc.append(line['meta_test_supp_acc1'])
            query_acc.append(line['meta_test_acc1'])

    return supp_acc, query_acc


def frozen_pretrained_transformer():
    plt.rcParams["font.size"] = 25
    base_path = '/home/max/Documents/pretrain_frozen_transformer_logs/gpt2'

    # datasets = ['mnist', 'cifar10', 'cifar100']
    # titles = ['MNIST', "CIFAR-10", "CIFAR-100"]
    datasets = ['miniImageNet', 'tieredimagent']
    titles = ['MiniImageNet', "TieredImageNet"]
    methods = ["frozen_attn_mlp", "fully_ft", "scratch_frozen_attn_mlp", "scratch"]
    colors = ['green', 'blue', 'orange', 'brown']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(33, 16))

    for i, dataset in enumerate(datasets):
        ax = axes[0, i]
        ax.set_title(f"{titles[i]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Accuracy")

        ax2 = axes[1, i]
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Training Accuracy")

        for j, method in enumerate(methods):
            if i == 1 and j == 2: continue

            if i == 0 and j == 0:
                method = 'frozen_attn_nlp'

            path_ = f"{base_path}/{dataset}/{method}/log.txt"
            test_top1, train_top1 = process_logfile(path_)
            ax.plot(test_top1, color=colors[j], linestyle='-', lw=2)
            ax2.plot(train_top1, color=colors[j], linestyle='-', lw=2)
            # print(len(test_top1), dataset, method)

        # if i < 2:
        #     fpt_ben, n = benchmark[i]
        #     x = [fpt_ben for _ in range(n)]
        #     ax.plot(x, color='red', linestyle='--', lw=2)

        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    labels = ["Pretrained (Frozen Attn & MLP)", "Pretrained (Fully-Finetuned)",
              "Random Init (Frozen Attn & MLP)", "Random Init (Fully-optimized)"
              ]

    fig.legend(handles=axes[0, 0].lines,
               labels=labels,
               loc="lower center",
               fancybox=False,
               ncol=4,
               bbox_to_anchor=(0.01, 0.0, 0.9, 0.02),
               frameon=False,
               borderaxespad=0.,
               mode='expand')

    plt.savefig(f"plots/fpt-mini-tiered.pdf", bbox_inches='tight')


def bert_compare():
    plt.rcParams["font.size"] = 25
    base_path = '/home/max/Documents/fs-learning/google'
    models = ["bert_uncased_L-6_H-512_A-8", "bert_uncased_L-8_H-512_A-8"]
    short_path = []
    # datasets = ['miniImageNet', 'tieredimagent']
    titles = ['Bert (6 Blocks)', "Bert (8 Blocks)"]

    methods = ["frozen_attn_frozen_mlp_1e-05",
               "scratch_1e-05"
               ]

    colors = ['green', 'brown', 'orange', 'brown']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(33, 16))

    for i, model in enumerate(models):
        ax = axes[0, i]
        ax.set_title(f"{titles[i]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Meta Test Accuracy")

        ax2 = axes[1, i]
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Base Training Accuracy")

        for j, method in enumerate(methods):
            if i == 0:
                in_net = "FC"
            else:
                in_net = "CNN"

            path_ = f"{base_path}/{model}/tieredImageNet/5-way/1-shot/FLT/{in_net}/{method}/log.txt"
            test_top1, train_top1 = process_logfile(path_)
            x = np.arange(0, len(test_top1)) * 5
            ax.plot(x, test_top1, color=colors[j], linestyle='-', lw=2)
            ax2.plot(x, train_top1, color=colors[j], linestyle='-', lw=2)

        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    labels = ["Pretrained (Frozen Attn & MLP)",
              # "Pretrained (Fully-Finetuned)",
              # "Random Init (Frozen Attn & MLP)",
              "Random Init (Fully-optimized)"
              ]

    fig.legend(handles=axes[0, 0].lines,
               labels=labels,
               loc="lower center",
               fancybox=False,
               ncol=4,
               bbox_to_anchor=(0.1, 0.0, 0.8, 0.02),
               frameon=False,
               borderaxespad=0.,
               mode='expand')

    plt.savefig(f"plots/tiered-imagenet-model-compare.pdf", bbox_inches='tight')


def bert_small_few_shot():
    plt.rcParams["font.size"] = 25
    base_path = '/home/max/Documents/fs-learning/google/bert_uncased_L-6_H-512_A-8/tieredImageNet/5-way/1-shot/FLT/FC/'

    methods = [
        "frozen_attn_frozen_mlp_1e-05_cls_tkn",
        "fully_finetune_1e-05_cls_tkn",
        "scratch_frozen_attn_frozen_mlp_1e-05_cls_tkn",
        "scratch_1e-05_cls_tkn"

    ]

    # colors = ['purple', 'blue', 'orange', 'brown']
    colors = ['green', 'blue', 'orange', 'brown']
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(32, 12))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Base Training Accuracy")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Meta Test Accuracy")

    for i, model in enumerate(methods):
        path_ = f"{base_path}/{model}/log.txt"
        test_top1, train_top1 = process_logfile(path_)
        x = np.arange(0, len(test_top1), step=1) * 5
        ax2.plot(x, test_top1, color=colors[i], linestyle='-', lw=2)
        ax1.plot(x, train_top1, color=colors[i], linestyle='-', lw=2)

        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    labels = ["Pretrained (Frozen Attn & MLP)", "Pretrained (Fully-Finetuned)",
              "Random Init (Frozen Attn & MLP)", "Random Init (Fully-Optimized)"
              ]

    fig.legend(handles=ax1.lines,
               labels=labels,
               loc="lower center",
               fancybox=False,
               ncol=4,
               bbox_to_anchor=(0.01, -0.02, 0.9, 0.02),
               frameon=False,
               borderaxespad=0.,
               mode='expand')

    plt.savefig(f"plots/tiered-imagenent-bert-small.pdf", bbox_inches='tight')


def bert_large_few_shot():
    plt.rcParams["font.size"] = 25
    base_path = '/home/max/Documents/fs-learning/google/' \
                'bert_uncased_L-12_H-768_A-12/tieredImageNet/5-way/1-shot/FLT/FC/'

    methods = [
        # "frozen_attn_frozen_mlp_3e-05_cls_tkn",
        # "frozen_attn_frozen_mlp_3e-05_cls_tkn_bias",
        # "frozen_attn_frozen_mlp3e-05_cls_bias_diff_lr",
        "fully_finetune_3e-05_cls_tkn",
        "scratch_3e-05_cls_tkn"
    ]

    # colors = ['purple', 'blue', 'orange', 'brown']
    colors = ['green', 'blue', 'orange', 'brown']
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(32, 12))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Base Training Accuracy")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Meta Test Accuracy")

    for i, model in enumerate(methods):
        path_ = f"{base_path}/{model}/log.txt"
        test_top1, train_top1 = process_logfile(path_)
        x = np.arange(0, len(test_top1), step=1) * 5
        ax2.plot(x, test_top1, color=colors[i], linestyle='-', lw=2)
        ax1.plot(x, train_top1, color=colors[i], linestyle='-', lw=2)

        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    labels = [
              "Pretrained (Fully-Finetuned)",
              "Scratch (Fully-Optimized)"
              ]

    fig.legend(handles=ax1.lines,
               labels=labels,
               loc="lower center",
               fancybox=False,
               ncol=2,
               bbox_to_anchor=(0.2, -0.04, 0.6, 0.02),
               frameon=False,
               borderaxespad=0.,
               mode='expand')

    plt.savefig(f"plots/tiered-imagenet-bert-large.pdf", bbox_inches='tight')


def vit_few_shot():
    plt.rcParams["font.size"] = 25
    base_path = '/home/max/Documents/fs-learning/vit-base/tieredImageNet/5-way/1-shot/FLT/FC/'

    methods = ["fully_finetune_3e-05",
               "fully_finetune_0.0005",
               "scratch_3e-05",
               "scratch_0.0005"
               ]

    # colors = ['purple', 'blue', 'orange', 'brown']
    colors = ['g-', 'g--', 'r', 'r--']
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(32, 12))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Base Training Accuracy")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Meta Test Accuracy")

    for i, model in enumerate(methods):
        path_ = f"{base_path}/{model}/224_cls_tkn/log.txt"
        test_top1, train_top1 = process_logfile(path_)
        x = np.arange(0, len(test_top1), step=1) * 5
        ax2.plot(x, test_top1, colors[i], lw=2)
        ax1.plot(x, train_top1, colors[i], lw=2)

        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    labels = ["IMNET Pretrained (LR_3e-5)",
              "IMNET Pretrained (LR_5e-4",
              "Random Init (LR_3e-5)",
              "Random Init (LR_5e-4)"
              ]

    fig.legend(handles=ax1.lines,
               labels=labels,
               loc="lower center",
               fancybox=False,
               ncol=2,
               bbox_to_anchor=(0.2, -0.04, 0.6, 0.02),
               frameon=False,
               borderaxespad=0.,
               mode='expand')

    plt.savefig(f"plots/tiered-imagenet-vit-base.pdf", bbox_inches='tight')


def bert_small_ncls():
    plt.rcParams["font.size"] = 25
    base_path = '/home/max/Documents/fs-learning/google/bert_uncased_L-6_H-512_A-8/tieredImageNet/5-way/1-shot/FLT/FC/'

    methods = ["frozen_attn_frozen_mlp_1e-03",
               "frozen_attn_frozen_mlp_1e-05",
               "scratch_1e-05",
               # "scratch_1e-05_cls_tkn"
               ]

    # colors = ['purple', 'blue', 'orange', 'brown']
    colors = ['green', 'green', 'brown']
    ls = ['--', '-', '-']
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(32, 12))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Base Training Accuracy")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Meta Test Accuracy")

    for i, model in enumerate(methods):
        path_ = f"{base_path}/{model}/log.txt"
        test_top1, train_top1 = process_logfile(path_)
        x = np.arange(0, len(test_top1), step=1) * 5
        ax2.plot(x, test_top1, color=colors[i], linestyle=ls[i], lw=2)
        ax1.plot(x, train_top1, color=colors[i], linestyle=ls[i], lw=2)

        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    labels = ["Pretrained (Frozen Attn & MLP)-1e-3",
              "Pretrained (Frozen Attn & MLP)-1e-5",
              "Random Init (Fully-optimized)-1e-5"
              ]

    fig.legend(handles=ax1.lines,
               labels=labels,
               loc="lower center",
               fancybox=False,
               ncol=4,
               bbox_to_anchor=(0.01, -0.02, 0.9, 0.02),
               frameon=False,
               borderaxespad=0.,
               mode='expand')

    plt.savefig(f"plots/tiered-imagenent-lr-rate.pdf", bbox_inches='tight')


def bert_small_x():
    plt.rcParams["font.size"] = 25
    base_path = '/home/max/Documents/fs-learning/google/'
    other_p = "/tieredImageNet/5-way/1-shot/FLT/FC/frozen_attn_frozen_mlp_1e-03/log.txt"
    models = ["bert_uncased_L-4_H-256_A-4",
              "bert_uncased_L-6_H-512_A-8",
              ]

    # methods = ["frozen_attn_frozen_mlp",
    #            "fully_finetune",
    #            "scratch",
    #            # "scratch_1e-05_cls_tkn"
    #            ]

    # colors = ['purple', 'blue', 'orange', 'brown']
    colors = ['red', 'green', 'brown']
    ls = ['--', '-', '-']
    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(19, 12))

    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Base Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Meta Test Accuracy")

    for i, model in enumerate(models):
        path_ = f"{base_path}/{model}/{other_p}"
        test_top1, train_top1 = process_logfile(path_)
        x = np.arange(0, len(test_top1), step=1) * 5
        ax2.plot(x, test_top1, color=colors[i], linestyle=ls[i], lw=2)
        # ax1.plot(x, train_top1, color=colors[i], linestyle=ls[i], lw=2)

        # ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    labels = ["Pretrained (Frozen Attn & MLP, L = 4, H = 256)",
              "Pretrained (Frozen Attn & MLP, L = 6, H = 512)",
              # "Random Init (Fully-optimized)-1e-3"
              ]
    # ax2.legend(labels)
    fig.legend(handles=ax2.lines,
               labels=labels,
               loc="lower center",
               fancybox=False,
               ncol=4,
               bbox_to_anchor=(0.00, -0.02, 1.0, 0.02),
               frameon=False,
               borderaxespad=0.,
               mode='expand')

    plt.savefig(f"plots/tiered-imagenet-compare-frozen.pdf", bbox_inches='tight')


def meta_results():
    base_path = '/home/max/Documents/few-shot-logs/google'

    models = ['bert_uncased_L-12_H-256_A-4', 'bert_uncased_L-12_H-768_A-12']
    # models = ['bert_uncased_L-12_H-256_A-4']
    model_common = ['Bert-Small', "Bert-Base"]
    methods = ["scratch", "freeze-attn-mlp", "finetune-all"]
    datasets = ["FC100", "miniImageNet"]
    colors = ['red', 'green', 'blue']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(28, 18))

    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            ax.set_title(f"{model_common[i]},  {dataset}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")

            # if i == 1 and j == 1:
            #     continue

            path_ = f"{base_path}/{model}/{dataset}/5-ways/1-shots"

            for k, method in enumerate(methods):
                test_top1, meta_top1 = process_logfile(f"{path_}/{method}/log.txt")
                ax.plot(test_top1, color=colors[k], linestyle='-', lw=3)
                ax.plot(meta_top1, color=colors[k], linestyle='--', lw=3)
                print(max(meta_top1), model, method, dataset)

            ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    labels = ["Base-Val(Scratch)", "Meta-Val (Scratch)",
              "Base-Val (Frozen Attn & MLP)", "Meta-Val (Frozen Attn & MLP)",
              "Base-Val (Finetuned)", "Meta-Val (Finetuned)"
              ]

    fig.legend(handles=axes[0, 0].lines,
               labels=labels,
               loc="lower center",
               fancybox=False,
               ncol=3,
               bbox_to_anchor=(0., 0, 0.95, 0.02),
               frameon=False,
               borderaxespad=0.,
               mode='expand')

    plt.savefig(f"plots/meta_results.pdf", bbox_inches='tight')


def bert_frozen():
    plt.rcParams["font.size"] = 25
    base_path = '/home/max/Documents/fs-learning/google/bert_uncased_L-12_H-768_A-12/tieredImageNet' \
                '/5-way/1-shot/Encoder/CNN'

    methods = [
        "pretrain_frozen_frozen_attn_frozen_mlp",
        "scratch_frozen_attn_frozen_mlp",
         "fully_finetune"
    ]

    colors = ['green', 'blue', 'orange', 'red']
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(32, 13))

    ax1.set_ylabel("Support Accuracy")
    ax1.set_xlabel("Epoch")

    ax2.set_ylabel("Query Accuracy")
    ax2.set_xlabel("Epoch")
    fmt = ['-', '--', '-']
    for i, model in enumerate(methods):
            path_ = f"{base_path}/{model}/log.txt"
            supp_acc, query_acc = meta_log(path_)
            x = np.arange(0, len(supp_acc)) * 2
            ax1.plot(x, supp_acc,  color=colors[i], linestyle=fmt[i], lw=2)
            ax2.plot(x, query_acc, color=colors[i], linestyle=fmt[i], lw=2)
            # print(np.mean(query_acc))

    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    labels = ["Pretrained Embeddings", "Random Initialization", "Fully Finetuned"]

    fig.legend(handles=ax1.lines,
               labels=labels,
               loc="lower center",
               fancybox=False,
               ncol=4,
               bbox_to_anchor=(0.1, -0.01, 0.6, 0.04),
               frameon=False,
               borderaxespad=0.,
               mode='expand')

    plt.savefig(f"plots/impact-of-pretraining-2.pdf", bbox_inches='tight')


def meta_results_model():
    base_path = '/home/max/Documents/few-shot-logs/google/bert_uncased_L-12_H-768_A-12'

    # models = ['bert_uncased_L-12_H-256_A-4', 'bert_uncased_L-12_H-768_A-12']
    # models = ['bert_uncased_L-12_H-256_A-4']

    # model_common = ['Bert-Small', "Bert-Base"]

    methods = ["scratch", "freeze-attn-mlp", "finetune-all"]
    datasets = ["FC100", "miniImageNet"]
    colors = ['red', 'green', 'blue']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))

    for i, dataset in enumerate(datasets):
        ax1 = axes[i, 0]
        ax1.set_title(f"{dataset}, Base Validation")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")

        ax2 = axes[i, 1]
        ax2.set_title(f"{dataset}, Meta Validation")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")

        path_ = f"{base_path}/{dataset}/5-ways/1-shots"

        for k, method in enumerate(methods):
            test_top1, meta_top1 = process_logfile(f"{path_}/{method}/log.txt")
            ax1.plot(test_top1, color=colors[k], linestyle='-', lw=3)
            ax2.plot(meta_top1, color=colors[k], linestyle='--', lw=3)

        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    labels = ["Scratch",
              "Frozen Attn & MLP",
              "Fully-Finetuned",
              ]

    axes[0, 1].legend(labels)
    # fig.legend(handles=axes[0, 0].lines,
    #            labels=labels,
    #            loc="lower center",
    #            fancybox=False,
    #            ncol=3,
    #            bbox_to_anchor=(0.2, 0.01, 0.6, 0.),
    #            frameon=False,
    #            borderaxespad=0.,
    #            mode='expand')

    plt.savefig(f"plots/bert-base-meta_results.pdf", bbox_inches='tight')


def meta_results_five_shot():
    base_path = '/home/max/Documents/few-shot-logs/google/bert_uncased_L-12_H-256_A-4'

    # models = ['bert_uncased_L-12_H-256_A-4', 'bert_uncased_L-12_H-768_A-12']
    # models = ['bert_uncased_L-12_H-256_A-4']

    # model_common = ['Bert-Small', "Bert-Base"]

    methods = ["scratch", "finetune-all"]
    # datasets = ["FC100", "miniImageNet"]
    colors = ['red', 'blue']
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(30, 20))

    ax1.set_title(f"MiniImageNet, Base Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")

    ax2.set_title(f"MiniImageNet, Meta Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")

    path_ = f"{base_path}/miniImageNet/5-ways/5-shots"

    for k, method in enumerate(methods):
        test_top1, meta_top1 = process_logfile(f"{path_}/{method}/log.txt")
        ax1.plot(test_top1, color=colors[k], linestyle='-', lw=3)
        ax2.plot(meta_top1, color=colors[k], linestyle='--', lw=3)

    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    labels = ["Scratch",
              "Fully-Finetuned",
              ]

    ax1.legend(labels)
    # fig.legend(handles=axes[0, 0].lines,
    #            labels=labels,
    #            loc="lower center",
    #            fancybox=False,
    #            ncol=3,
    #            bbox_to_anchor=(0.2, 0.01, 0.6, 0.),
    #            frameon=False,
    #            borderaxespad=0.,
    #            mode='expand')

    plt.savefig(f"plots/bert-small-meta_results-5-shot.pdf", bbox_inches='tight')


if __name__ == '__main__':
    # frozen_pretrained_transformer()
    #
    # bert_small_few_shot()
    # bert_large_few_shot()
    bert_frozen()
    # vit_few_shot()
    # bert_small_ncls()
    # bert_compare()
    # bert_small_x()
    # meta_results()
    # meta_results_model()
    # meta_results_five_shot()

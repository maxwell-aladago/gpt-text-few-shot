from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

import os

from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


def pil_loader(img_path):
    with open(img_path, 'rb') as img_f:
        img = Image.open(img_f)
        return img.convert("RGB")


class TieredImageNet(Dataset):
    def __init__(self, args, partition='train', word_embeddings="./data/gpt3_txt_bert_uncased_L-8_H-512_A-8_embed"):
        super(Dataset, self).__init__()
        self.imnet_root = args.data_root
        self.meta_file = np.load(f"./data/{partition}_meta.npy", allow_pickle=True)

        self.we_meta = torch.load(f"{word_embeddings}_{partition}.pth", map_location='cpu')

        self.transform = build_transform('train' in partition, args)
        self.loader = pil_loader

    def __getitem__(self, item):
        sample = self.meta_file[item]
        img = self.loader(os.path.join(self.imnet_root, sample['imnet_cat'], sample['img_name']))
        img = self.transform(img)
        target = sample['label']

        # word embeddings
        word_embed = self.we_meta[target]
        return img, word_embed, target

    def __len__(self):
        return len(self.meta_file)


class MetaTieredImageNet(TieredImageNet):
    def __init__(self, args, partition='test', word_embeddings="/data/gpt3_txt_bert_uncased_L-8_H-512_A-8_embed",
                 num_ways=5, num_shots=1, num_queries=5,
                 augmentations=5, num_test_runs=600, fix_seed=True):
        super(MetaTieredImageNet, self).__init__(args, partition, word_embeddings)

        self.fix_seed = fix_seed
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.augmentations = augmentations

        self.num_test_runs = num_test_runs

        self.train_transform = build_transform(True, args)
        self.test_transform = build_transform(False, args)

        self.data = {}
        for i in range(len(self.meta_file)):
            sample = self.meta_file[i]
            label = sample['label']
            if label not in self.data:
                self.data[label] = []

            self.data[label].append((sample['imnet_cat'], sample['img_name']))

        self.classes = list(self.data.keys())

    def read_imgs(self, idx):
        images = []
        for i in range(len(idx)):
            img_cat, img_name = idx[i]
            path_f = os.path.join(self.imnet_root, img_cat, img_name)
            images.append(self.loader(path_f))

        return images

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)

        cls_sampled = np.random.choice(self.classes, self.num_ways, False)

        supp_xs = []
        supp_ys = []
        query_xs = []
        query_ys = []

        for idx, cls in enumerate(cls_sampled):
            # get all the images belonging to a class
            imgs = np.array(self.data[cls])

            # select the indices of a subset of the support indices
            cls_ids = np.arange(len(imgs))
            supp_ids_sampled = np.random.choice(cls_ids, self.num_shots, False)
            sampled_supp = imgs[supp_ids_sampled]
            supp_xs.append(sampled_supp)
            supp_ys.append([idx] * self.num_shots)

            # select a disjoint set of indices for the queries
            query_xs_ids = np.setxor1d(cls_ids, supp_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.num_queries, False)
            sampled_qs = imgs[query_xs_ids]
            query_xs.append(sampled_qs)
            query_ys.append([idx] * self.num_queries)

        supp_xs, supp_ys = np.array(supp_xs), np.array(supp_ys)
        query_xs, query_ys = np.array(query_xs), np.asarray(query_ys)

        t_size = query_xs.shape[-1]
        supps, queries = (self.num_ways * self.num_shots), (self.num_ways * self.num_queries)
        query_xs = query_xs.reshape((queries, t_size))
        query_ys = query_ys.reshape(queries,)

        supp_xs = supp_xs.reshape((supps, t_size))
        supp_ys = supp_ys.reshape(supps,)
        if self.augmentations > 1:
            supp_xs = np.tile(supp_xs, (self.augmentations, 1))
            supp_ys = np.tile(supp_ys, self.augmentations)

        supp_xs = self.read_imgs(supp_xs)
        query_xs = self.read_imgs(query_xs)

        supp_xs = torch.stack(list(map(lambda x: self.train_transform(x), supp_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x), query_xs)))

        return supp_xs, supp_ys, query_xs, query_ys

    def __len__(self):
        return self.num_test_runs


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )

        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    # t = [lambda x: Image.fromarray(x)]
    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np


class SIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None

    def forward(self, img_embed, text_embed, labels):
        img_embed = F.normalize(img_embed, dim=-1, p=2)

        local_batch_size = img_embed.size(0)

        k_a, k_b, k_l = utils.all_gather_batch_with_grad([img_embed, text_embed, labels])

        mask = labels.reshape(-1, 1).float() @ k_l.reshape(1, -1).float()
        mask = (mask == torch.square(k_l)).long()
        mask.fill_diagonal_(0)
        mask = mask * 1e9

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=img_embed.device
            )

            total_batch_size = local_batch_size * utils.get_world_size()
            self.masks = F.one_hot(self.labels, total_batch_size) * 1e9

            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(img_embed, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - self.masks - mask
        logits_bb = torch.matmul(text_embed, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks - mask
        logits_ab = (torch.matmul(img_embed, k_b.transpose(0, 1)) / self.tau) - mask
        logits_ba = (torch.matmul(text_embed, k_a.transpose(0, 1)) / self.tau) - mask

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'acc': acc}


class MetaSIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, img_embed, text_embed, labels):
        img_embed = F.normalize(img_embed, dim=-1, p=2)

        local_batch_size = img_embed.size(0)

        k_a, k_b = utils.all_gather_batch_with_grad([img_embed, text_embed])

        labels = local_batch_size * utils.get_rank() + labels

        total_batch_size = local_batch_size * utils.get_world_size()

        masks = F.one_hot(labels, total_batch_size) * 1e9

        logits_aa = torch.matmul(img_embed, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - masks
        logits_bb = torch.matmul(text_embed, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - masks
        logits_ab = torch.matmul(img_embed, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(text_embed, k_a.transpose(0, 1)) / self.tau

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'acc': acc}


class RankingLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, image_embed, text_embed, labels):
        local_batch_size = image_embed.size(0)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all, labels_all = \
            utils.all_gather_batch([image_embed, text_embed, labels])

        mask = labels.reshape(-1, 1).float() @ labels_all.reshape(1, -1).float()
        mask = (mask == torch.square(labels)).long()
        mask.fill_diagonal_(0)
        mask = mask * 1e9

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )

            # total_batch_size = local_batch_size * utils.get_world_size()
            self.last_local_batch_size = local_batch_size

        # print(image_embed.shape, text_embed_all.shape, self.labels.shape)

        logits_per_image = (image_embed @ text_embed_all.t()) - mask
        logits_per_text = (text_embed @ image_embed_all.t()) - mask

        loss_a = F.multi_margin_loss(logits_per_image, self.labels, margin=self.margin)
        loss_b = F.multi_margin_loss(logits_per_text, self.labels, margin=self.margin)
        loss = (loss_a + loss_b) / 2.0

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'acc': acc}


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, outputs):
        image_embed = outputs['image_embed']
        text_embed = outputs['text_embed']
        # logit_scale = outputs['logit_scale']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all = \
            utils.all_gather_batch([image_embed, text_embed])

        # cosine similarity as logits
        logits_per_image = self.logit_scale * (image_embed @ text_embed_all.t())
        logits_per_text = self.logit_scale * (text_embed @ image_embed_all.t())

        loss = (F.cross_entropy(logits_per_image, self.labels) +
                F.cross_entropy(logits_per_text, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'acc': acc}

# if __name__ == '__main__':
#     # x = torch.randn(10, 20)
#     # y = torch.randn(10, 20)
#     #
#     loss = SIMCLRLoss()
#     z = loss(x, y)

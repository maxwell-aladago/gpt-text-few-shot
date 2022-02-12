# import a pretrained bert-model
# grab the various templates for imagenet classes
# for each sample in our meta-file:
# grab the list of class names:
# * for each name in that list
# * insert it into all templates
# * tokenize the all sentences
# * compute the appropriate bert embeddings for each class using bert
# * map its representation to the integer class-labels

import os
import json
import numpy as np
import argparse
import torch.nn.functional as F
import torch

from transformers import BertModel, BertTokenizer


def process(pt_model="bert_uncased_L-8_H-512_A-8", is_train=False):
    tokenizer = BertTokenizer.from_pretrained(f"google/{pt_model}")
    model = BertModel.from_pretrained(f"google/{pt_model}").to('cuda')
    model.eval()

    key = "train" if is_train else "test"

    with open(f"./data/gpt3_text_{key}.json") as f:
        gpt3_gen_text = json.load(f)

    with open("./data/ids_2_names.json") as f:
        ids_2_names = json.load(f)[key]

    word_embed = {}
    tensor_out = []
    for label in range(len(ids_2_names)):
        class_name = ids_2_names[str(label)]

        if isinstance(class_name, list):
            class_name = class_name[0]

        texts = gpt3_gen_text[class_name]

        # some entries are empty strings (because they were duplicates and got removed)
        texts = [t for t in texts.values() if len(t) > 0]
        texts = tokenizer(texts, return_tensors='pt', padding='longest')
        texts = texts.to('cuda')
        outputs = model(**texts)

        class_embeddings = outputs.last_hidden_state[:, 0]  # select class token

        class_embeddings = F.normalize(class_embeddings, dim=-1)
        class_embeddings = class_embeddings.mean(dim=0)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        class_embeddings = class_embeddings.detach().cpu()

        word_embed[label] = class_embeddings.tolist()
        tensor_out.append(class_embeddings)

    tensor_out = torch.stack(tensor_out, dim=0)
    print(f"Processed {key} set of shape", tensor_out.shape)

    file_name = f"./data/gpt3_txt_{pt_model}_embed_{key}"
    torch.save(tensor_out, file_name + ".pth" )
    np.save(f"{file_name}.npy", np.array([word_embed]))


def ids_to_names():
    # process train set
    train_meta_file = np.load("./data/train_meta.npy", allow_pickle=True)
    train_set = {}
    for sample in train_meta_file:
        class_id = sample['label']

        if class_id not in train_set:
            train_set[class_id] = sample['synet_name']

    # process test-set
    test_meta_file = np.load("./data/test_meta.npy", allow_pickle=True)
    test_set = {}
    for sample in test_meta_file:
        class_id = sample['label']
        if class_id not in test_set:
            test_set[class_id] = sample['synet_name']

    cls_ids_2_names = {"train": train_set, "test": test_set}
    with open("./data/ids_2_names.json", 'w') as f:
        f.write(json.dumps(cls_ids_2_names, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_model", default='bert_uncased_L-8_H-512_A-8', type=str)

    args = parser.parse_args()

    # ids_to_names()
    process(pt_model=args.pt_model, is_train=True)

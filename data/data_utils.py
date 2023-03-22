import json
import torch
import numpy as np
from transformers import AutoTokenizer

CHECK_POINT = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(CHECK_POINT)

def load_labels(labels_file: str):
    labels = None
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()
    id2label = {ids: label for ids, label in enumerate(labels)}
    label2id = {label: ids for ids, label in enumerate(labels)}
    return id2label, label2id

id2label, label2id = load_labels('./data/datasets/labels.txt')

def load_data(path: str) -> list:
    res = []
    with open(path, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f.read().split('\n')):
            if not line:
                break
            obj = json.loads(line)
            sentence = obj['text']
            label = obj['label']
            res.append({'sentence': sentence, 'labels': label})
    return res


def collate_fn(batch_samples):
    batch_sentences, batch_tags = [], []
    for sample in batch_samples:
        batch_sentences.append(sample['sentence'])
        batch_tags.append(sample['labels'])

    batch_inputs = tokenizer(batch_sentences, padding=True,
                             truncation=True, return_tensors='pt')
    batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)

    for idx, sentence in enumerate(batch_sentences):
        encoding = tokenizer(sentence, truncation=True)
        batch_label[idx][0] = 0
        batch_label[idx][len(encoding.tokens())-1:] = 0

        for start, end, tag in batch_tags[idx]:
            token_start = encoding.char_to_token(start)
            token_end = encoding.char_to_token(end-1)
            batch_label[idx][token_start] = label2id[f'B-{tag}']
            try:
                batch_label[idx][token_start +
                                 1:token_end+1] = label2id[f'I-{tag}']
            except Exception as e:
                print(idx, sentence)

    return batch_inputs, torch.LongTensor(batch_label)

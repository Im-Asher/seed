import json
import torch
import numpy as np
from transformers import AutoTokenizer


id2label = {0: 'O', 1: 'B-SOFT', 2: 'I-SOFT', 3: 'B-VER', 4: 'I-VER'}
label2id = {'O': 0, 'B-SOFT': 1, 'I-SOFT': 2, 'B-VER': 3, 'I-VER': 4}

CHECK_POINT = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(CHECK_POINT)


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
            token_end = encoding.char_to_token(end)
            batch_label[idx][token_start] = label2id[f'B-{tag}']
            batch_label[idx][token_start+1:token_end] = label2id[f'I-{tag}']

    return batch_inputs, torch.LongTensor(batch_label)

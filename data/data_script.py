import pandas as pd
import json
from tqdm import tqdm
from math import ceil
from sklearn.model_selection import train_test_split


def jsonl_to_json(filepath: str, to_dir: str, file_name='cve-500.json'):
    input_data = None
    res = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in tqdm(enumerate(f.read().split('\n'))):
            if not line:
                break
            obj = json.loads(line)
            id_ = obj['id']
            sentence = obj['text']
            labels = obj['label']
            res.append({'id': id_,
                        'sentence': sentence,
                        'labels': [{"start": label[0],
                                    "end":label[1],
                                    "label":label[2]} for label in labels]})
    to_path = to_dir + '/'+file_name
    with open(to_path, 'wt', encoding='utf-8') as f:
        json.dump(res, f)
    print("Convert finished!")


def split_data(filepath, to_dir: str, file_name: str, split_rate=0.8):
    samples = None

    with open(filepath, 'r', encoding='utf-8') as f:
        samples = f.readlines()

    if split_rate >= 1:
        raise Exception("Error split rate value(>=100%)")

    n_samples = len(samples)
    n_train = ceil(n_samples * split_rate)
    n_test = n_samples - n_train

    train = samples[:n_train]
    test = samples[n_train:]

    train_file = to_dir+"/"+file_name+"-train.jsonl"
    test_file = to_dir+"/"+file_name+"-test.jsonl"
    
    with open(train_file,'w',encoding='utf-8') as f:
        f.writelines(train)
    
    with open(test_file,'w',encoding='utf-8') as f:
        f.writelines(test)

if __name__ == "__main__":
    split_data(filepath='data\datasets\cve_description\cve-500.jsonl',
                  to_dir='data\datasets\cve_description',file_name='cve-500')

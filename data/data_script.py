import pandas as pd
import json
from tqdm import tqdm

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


if __name__ == "__main__":
    jsonl_to_json('data\datasets\cve_description\cve-500.jsonl',
                  to_dir='data\datasets\cve_description')

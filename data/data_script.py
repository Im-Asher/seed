import pandas as pd
import json
from tqdm import tqdm

output_path = '.\cve_description.txt'


def json_to_text(json_path: str):
    input_data = None
    with open(json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    res = []
    for item in input_data:
        res.append(item['description'])
    with open(output_path, 'a', encoding='utf-8') as f:
        for desc in res:
            f.write(desc)
            f.write('\n')
        
    print('finished!')


if __name__ == "__main__":
    json_to_text('data/cve_description.json')

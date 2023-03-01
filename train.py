import torch
import json
import numpy as np

from pipeline_provider import BertCrfPipeline
from model_provider import BertCrfForNer
from config_provider import BertCrfConfig
from transformers.pipelines import SUPPORTED_TASKS
from transformers import AutoTokenizer
from transformers import pipeline

from datasets import Dataset, load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments

id2label = {0: 'O', 1: 'B-SOFT', 2: 'I-SOFT', 3: 'B-VER', 4: 'I-VER'}
label2id = {'O': 0, 'B-SOFT': 1, 'I-SOFT': 2, 'B-VER': 3, 'I-VER': 4}


def data_generator():
    filepath = "./data/datasets/cve_description/cve-500.jsonl"
    with open(filepath, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f.read().split('\n')):
            if not line:
                break
            obj = json.loads(line)
            id_ = obj['id']
            sentence = obj['text']
            label = obj['label']
            yield {
                "id": id_,
                "sentence": sentence,
                "label": label
            }


dataset = load_dataset(path="./data/datasets/cve_description/")

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(batch_samples):
    batch_ids, batch_sentences, batch_tags = [], [], []

    batch_ids += (batch_samples['id'])
    batch_sentences += (batch_samples['sentence'])
    batch_tags += (batch_samples['labels'])

    batch_inputs = tokenizer(batch_sentences, padding='max_length',
                             truncation=True, return_tensors='pt')
    batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)

    batch_inputs['sentences'] = batch_sentences

    for idx, sentence in enumerate(batch_sentences):
        encoding = tokenizer(sentence, truncation=True)
        batch_label[idx][0] = 0
        batch_label[idx][len(encoding.tokens())-1:] = 0

        for t_idx in range(len(batch_tags[idx]['label'])):
            token_start = encoding.char_to_token(
                batch_tags[idx]['start'][t_idx])
            token_end = encoding.char_to_token(batch_tags[idx]['end'][t_idx]-1)
            batch_label[idx][token_start] = label2id[f'B-{ batch_tags[idx]["label"][t_idx]}']
            try:
                batch_label[idx][token_start +
                                 1:token_end+1] = label2id[f'I-{batch_tags[idx]["label"][t_idx]}']
            except Exception as e:
                print(idx, sentence)

    batch_inputs["labels"] = torch.LongTensor(batch_label)

    return batch_inputs

tokenized_datasets = dataset.map(
    tokenize_function, batched=True, batch_size=40)

small_train_dataset = tokenized_datasets["train"].shuffle(
    seed=42).select(range(400))
small_eval_dataset = tokenized_datasets["test"].shuffle(
    seed=42).select(range(100))

config = BertCrfConfig.from_pretrained('./model_cache/bert-crf')
model = BertCrfForNer.from_pretrained("./model_cache/bert-crf", config=config)

training_args = TrainingArguments(
    output_dir="model_cache/bert-crf", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
)

trainer.train()

trainer.save_model()

TASK_NAME = "sv-ner-task"
TASK_DEFINITION = {
    "impl": BertCrfPipeline,
    "tf": (),
    "pt": (BertCrfForNer,),
    "default": {},
    "type": "text",
}
SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION


sv_extractor = pipeline(
    task=TASK_NAME,
    model=model,
    tokenizer=tokenizer,
)

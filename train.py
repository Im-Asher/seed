import json
from pipeline_provider import BertCrfPipeline
from model_provider import BertCrfForNer
from transformers.pipelines import SUPPORTED_TASKS
from transformers import AutoTokenizer
from transformers import pipeline

from datasets import Dataset,load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments


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


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(
    seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(
    seed=42).select(range(1000))

model = AutoModelForMaskedLM.from_pretrained("bert-base-cased", num_labels=5)

training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
)

trainer.train()


TASK_NAME = "sv-ner-task"
TASK_DEFINITION = {
    "impl": BertCrfPipeline,
    "tf": (),
    "pt": (BertCrfForNer,),
    "default": {},
    "type": "text",
}
SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

model = BertCrfForNer.from_pretrained("")
tokenizer = AutoTokenizer.from_pretrained("")

sv_extractor = pipeline(
    task=TASK_NAME,
    model=model,
    tokenizer=tokenizer,
)

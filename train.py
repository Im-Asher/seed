from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

model = AutoModelForMaskedLM.from_pretrained("bert-base-cased", num_labels=5)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
)

trainer.train()


from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers.pipelines import SUPPORTED_TASKS
from pipeline_provider import BertCrfPipeline
from model_provider import BertCrfForNer

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
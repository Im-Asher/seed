import bentoml

from pipeline_provider import BertCrfPipeline
from model_provider import BertCrfForNer
from config_provider import BertCrfConfig
from transformers.pipelines import SUPPORTED_TASKS
from transformers import AutoTokenizer, AutoConfig, AutoModel

from transformers import AutoTokenizer 

NAME_OR_PATH = "model_cache/bert_crf_p90_SVRLF"
BERT_TOKENIZER = "model_cache/bert-base-uncased"
TASK_NAME = "sv-ner-task"

AutoConfig.register("bert-crf", BertCrfConfig)
AutoModel.register(BertCrfConfig, BertCrfForNer)

config = AutoConfig.from_pretrained(NAME_OR_PATH)
model = BertCrfForNer.from_pretrained(NAME_OR_PATH, config=config)
tokenizer = AutoTokenizer.from_pretrained(BERT_TOKENIZER, do_lower_case=True,model_max_length=512)

TASK_DEFINITION = {
    "impl": BertCrfPipeline,
    "tf": (),
    "pt": (AutoModel,),
    "default": {},
    "type": "text",
}

SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

generator = BertCrfPipeline(task=TASK_NAME, model=model, tokenizer=tokenizer)
saved_model = bentoml.transformers.save_model(
    "bert-crf", generator, task_name=TASK_NAME, task_definition=TASK_DEFINITION)

print(f'model saved:{saved_model}')

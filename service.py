import bentoml

from bentoml.io import Text,JSON
from model_provider import BertCrfForNer,BertCrfConfig
from transformers import AutoModel,AutoConfig

# Register custom model/config to AutoConfig & AutoModel
AutoConfig.register("bert-crf",BertCrfConfig)
AutoModel.register(BertCrfConfig,BertCrfForNer)

runner = bentoml.transformers.get("bert-crf:latest").to_runner()
svc = bentoml.Service("sv_ner_service@bert-crf@500")

@svc.api(input=Text,output=JSON())
def sv_extract(desc:str):
    pass

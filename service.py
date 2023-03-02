import bentoml

from bentoml.io import JSON
from model_provider import BertCrfForNer,BertCrfConfig
from transformers import AutoModel,AutoConfig

# Register custom model/config to AutoConfig & AutoModel
AutoConfig.register("bert-crf",BertCrfConfig)
AutoModel.register(BertCrfConfig,BertCrfForNer)

sv_runner = bentoml.transformers.get("bert-crf:latest").to_runner()
svc = bentoml.Service("sv_ner_service",runners=[sv_runner])

@svc.api(input=JSON(),output=JSON())
def extract(request: dict):
    res = {'status':0}
    if 'task' not in request.keys() or 'sv' == request.get('task'):
        results = sv_runner.run(request.get('samples'))
        for result in results:
            if result:
                res['status'] = 1
                res['results'] = results
                break
    return res
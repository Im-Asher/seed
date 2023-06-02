import bentoml

from bentoml.io import JSON
from model_provider import BertCrfForNer, BertCrfConfig
from transformers import AutoModel, AutoConfig
from utils.service_utils import Response
from utils.constant_enum import ResponseCode

# Register custom model/config to AutoConfig & AutoModel
AutoConfig.register("bert-crf", BertCrfConfig)
AutoModel.register(BertCrfConfig, BertCrfForNer)

sv_runner = bentoml.transformers.get("bert-crf:latest").to_runner()
svc = bentoml.Service("sv_ner_service", runners=[sv_runner])

ROUTE = "api/v1/ner/"


@svc.api(input=JSON(), output=JSON(), route=ROUTE + 'extract')
def extract(request: dict):
    res = Response()
    if "sv" == request.get("task"):
        try:
            sentences = [sentence for sentence in request.get(
                "samples") if len(sentence) > 0]
            if not sentences:
                raise Exception("Sentence is empty")
            results = sv_runner.run(sentences)
            for result in results:
                if result:
                    res.status = ResponseCode.Success.value
                    res.results = results
                    res.msg = "Success"
                    break
                else:
                    res.msg = "Can't extrcat entity information"
        except Exception as ex:
            res.msg = str(ex)
            res.status = ResponseCode.Exception.value
    else:
        res.msg = "Only support sv task ('task'='sv')"
    return res.__dict__

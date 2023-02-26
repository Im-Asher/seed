from transformers import Pipeline, TokenClassificationPipeline


class BertCrfPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = {}, {}, {}

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, text, **kwagrs):
        input_ids = self.tokenizer(text,
                                   return_offsets_mapping=True,
                                   return_tensors="pt",
                                   padding="max_length",
                                   truncation=True)
        input_ids["sentences"] = text
        input_ids["tokens"] = input_ids.tokens()
        return input_ids

    def _forward(self, model_inputs):
        sentences = model_inputs.pop("sentences")
        tokens = model_inputs.pop("tokens")
        offsets = model_inputs.pop("offsets_mapping")
        outputs = self.model(**model_inputs)
        self.model.crf.decode()
        return outputs

    def postprocess(self, model_outputs):   
        # decode -> BIO
        # output -> {entity\label\start\end}
        return model_outputs["logits"].softmax(-1).numpy()

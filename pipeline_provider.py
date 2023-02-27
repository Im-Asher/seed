import re
import torch

from transformers import Pipeline


class BertCrfPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = {}, {}, {}

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    # Data preprocess
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

        outputs = self.model(model_inputs)

        logits = outputs[0]
        mask = torch.tensor(model_inputs["attention_mask"], dtype=torch.uint8)
        tags = self.model.crf.decode(logits, mask)

        output = tags[0]
        output = [self.model.config.id2label[x] for x in output]

        return output, tokens, sentences, offsets[0]

    def postprocess(self, model_outputs):
        # decode -> BIO
        # output -> {entity\label\start\end}
        output, tokens, sentences, offsets = model_outputs

        idx = 0
        pred_label = []

        while idx < len(output):
            label = output[idx]
            if label != 'O':
                label = label[2:]
                start, end = offsets[idx]
                while idx + 1 < len(output) and output[idx + 1] == f'I-{label}':
                    _, end = offsets[idx+1]
                    idx += 1
                start, end = start.item(), end.item()
                word = sentences[start:end]
                if label == "VER":
                    word = self._convert_to_version_range(word)
                pred_label.append({
                    "entity_group": label,
                    "word": word,
                    "start": start,
                    "end": end
                })

            idx += 1

        return pred_label

    def _convert_to_version_range(self, version: str):
        pattern = re.compile(r'\d+\.(?:\d+\.)*\d+(?:-\w+)?')
        results = re.findall(pattern, version)
        v_range = version
        if len(results) > 0:
            if len(results) == 1:
                v_range = f'(,{results[0]}]'
            if len(results) == 2:
                v_range = f'[{results[0]},{results[1]}]'
            if len(results) > 2:
                t_range = ','.join(results)
                v_range = f'[{t_range[:-1]}]'
        return v_range

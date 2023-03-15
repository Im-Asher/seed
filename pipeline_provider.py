import re
import torch
import numpy as np

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
        offsets = model_inputs.pop("offset_mapping")

        outputs = self.model(**model_inputs)

        logits = outputs[0]

        probabilities = torch.nn.functional.softmax(
            logits, dim=-1)[0].cpu().numpy().tolist()

        mask = torch.tensor(model_inputs["attention_mask"], dtype=torch.uint8)
        tags = self.model.crf.decode(logits, mask)

        output = tags[0]
        output = [self.model.config.id2label[x] for x in output]

        return output, sentences, offsets[0], probabilities

    def postprocess(self, model_outputs):
        # decode -> BIO
        # output -> {entity\label\start\end}
        output,  sentences, offsets, probabilities = model_outputs

        idx = 0
        pred_label = []

        while idx < len(output):
            label, tag = output[idx], output[idx]

            if label != 'O':
                label = label[2:]
                start, end = offsets[idx]
                all_scores = [probabilities[idx]
                              [self.model.config.label2id[tag]]]
                while idx + 1 < len(output) and output[idx + 1] == f'I-{label}':
                    all_scores.append(
                        probabilities[idx+1][self.model.config.label2id[tag]])
                    _, end = offsets[idx+1]
                    idx += 1
                start, end = start.item(), end.item()
                word = sentences[start:end]
                score = np.mean(all_scores).item()
                if label == "VER":
                    word = self._convert_to_version_range(word)
                pred_label.append({
                    "entity_group": label,
                    "word": word,
                    "score": score,
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

    def _convert_to_version_range_v2(self, version: str):
        one_left = ['start']
        one_right = ['prior', 'before', 'through', 'to', 'up', 'earlier']
        two_nochange = ['prior', 'from', 'up', 'start', 'before']

        version_pattern = r'\d+\.\d+(?:\.\d+)?(?:\w+|-\w+)?'

        version_intervals = [(match.group(), match.start(), match.end())
                             for match in re.finditer(version_pattern, version)]
        
        if len(version_intervals) > 0:
            if len(version_intervals) == 1:
                for w in one_left:
                    s = version.find(w)
                    if s != -1:
                        return f"[{version_intervals.get(0)[0]},)","RANGE"
                for w in one_right:
                    s = version.find(w)
                    if s != -1:
                        return f"(,{version_intervals.get(0)[0]}]","RANGE"
                    
                return f'[{version_intervals.get(0)[0]}]',"LIST"
        
            if len(version_intervals) == 2:
                pass

            if len(version_intervals) >2:
                pass
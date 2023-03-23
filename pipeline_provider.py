import re
import torch
import numpy as np

from transformers import Pipeline

version_pattern = r'\d+\.\d+(?:\.\d+)?(?:\w+|-\w+)?'
NOT_SHOW_LABELS = ['B-FVERL', 'I-FVERL', 'B-FVERR', 'I-FVERR', 'O']
VERSION_LABELS = ['VERR', 'VERL']


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

            if label not in NOT_SHOW_LABELS:
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
                l_type = label
                if label in VERSION_LABELS:
                    word = self._convert_to_version_format(word, label)
                pred_label.append({
                    "entity_group": label,
                    "word": word,
                    "score": score,
                    "start": start,
                    "end": end
                })

            idx += 1

        return pred_label

    def _convert_to_version_format(self, entity: str, label: str):
        entity = entity.lower()

        if label == 'VERL':
            return self._convert_to_version_list(entity)
        if label == 'VERR':
            return self._convert_to_version_range(entity)

    def _convert_to_version_list(self, entity: str):
        version_intervals = [(match.group(), match.start(), match.end())
                             for match in re.finditer(version_pattern, entity)]

        versions = [v[0] for v in version_intervals]

        return versions

    def _convert_to_version_range(self, entity: str):
        one_left = ['start', 'from']
        one_right = ['prior', 'before', 'through', 'to', 'up', 'earlier']

        # special version convert to specific version (e.g 5.x->5.0)
        special_char_pattern = r'[/:*x]'
        special = re.compile(special_char_pattern, re.I)

        version_intervals = [(match.group(), match.start(), match.end())
                             for match in re.finditer(version_pattern, entity)]

        versions = [special.sub('0', v[0]) for v in version_intervals]

        versions = sorted(versions)

        if len(versions) < 1:
            s = entity.find('all')
            if s != -1:
                return f'[,]'
            return f'[]'

        if len(versions) == 1:
            for w in one_left:
                s = entity.find(w)
                if s != -1:
                    return self._comfirm_the_boundary(entity, f"'{versions[0]}',", 1)

            for w in one_right:
                s = entity.find(w)
                if s != -1:
                    return self._comfirm_the_boundary(entity, f",'{versions[0]}'", 1)

        if len(versions) == 2:
            return self._comfirm_the_boundary(entity, f"'{versions[0]}','{versions[1]}'", 2)
        if len(versions) >= 3:
            version_str = f"'{versions[0]}','{versions[-1]}'"
            return self._comfirm_the_boundary(entity, version_str, 3)

    def _comfirm_the_boundary(self, entity: str, versions: str, versions_size: int):
        including_key_word = ['include', 'includ', 'through']
        if versions_size < 1:
            return versions
        if versions_size == 1:
            for w in including_key_word:
                s = entity.find(w)
                if s != -1:
                    if versions.find(',') == 0:
                        return f"({versions}]"
                    else:
                        return f"[{versions})"
            return f"({versions})"
        if versions_size > 1:
            for w in including_key_word:
                s = entity.find(w)
                if s != -1:
                    return f"[{versions}]"
            return f"[{versions})"

    def _remove_duplicate_entity(entities: list):
        pass

    def _combine_version(entities: list):
        entities_size = len(entities)
        idx = 0
        results = []
        
        while idx < entities_size:
            software = None
            versions = []

            if entities[idx]['entity_group']=='SOFT':
                software = entities[idx]

            while idx+1 < entities_size and entities[idx+1]['entity_group'] != 'SOFT':
                versions.append(entities[idx+1])
                idx += 1

            results.append({'software':software,'versions':versions})
            
            idx += 1
        return results
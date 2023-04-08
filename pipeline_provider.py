import re
import torch
import nltk
import numpy as np

from transformers import Pipeline
from nltk.tokenize import sent_tokenize

version_pattern = r'\d+\.\d+(?:\.\d+)?(?:\w+|-\w+)?|\d+'
NOT_SHOW_LABELS = ['B-FVERL', 'I-FVERL', 'B-FVERR', 'I-FVERR', 'O']
VERSION_LABELS = ['VERR', 'VERL']


class BertCrfPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = {}, {}, {}

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    # Data preprocess
    def preprocess(self, text, **kwagrs):
        sent_text = sent_tokenize(text)
        input_ids = self.tokenizer(sent_text,
                                   return_offsets_mapping=True,
                                   return_tensors="pt",
                                   padding="max_length",
                                   truncation=True)
        input_ids["sentences"] = text
        input_ids["split_sent"] = sent_text
        input_ids["tokens"] = input_ids.tokens()
        return input_ids

    def _forward(self, model_inputs):
        sentences = model_inputs.pop("sentences")
        tokens = model_inputs.pop("tokens")
        offsets = model_inputs.pop("offset_mapping")
        sent_text = model_inputs.pop("split_sent")

        outputs = self.model(**model_inputs)

        logits = outputs[0]

        probabilities = torch.nn.functional.softmax(
            logits, dim=-1).cpu().numpy().tolist()

        mask = torch.tensor(model_inputs["attention_mask"], dtype=torch.uint8)
        tags = self.model.crf.decode(logits, mask)

        output = [[self.model.config.id2label[x] for x in tag]for tag in tags]

        return output, sentences, sent_text, offsets, probabilities

    def postprocess(self, model_outputs):
        # decode -> BIO
        # output -> {entity\label\start\end}
        output,  sentences, sent_text, offsets, probabilities = model_outputs

        pred_label = []
        for s_idx, sent in enumerate(output):
            idx = 0
            while idx < len(sent):
                label, tag = sent[idx], sent[idx]

                if label not in NOT_SHOW_LABELS:
                    label = label[2:]
                    start, end = offsets[s_idx][idx]
                    all_scores = [probabilities[s_idx][idx]
                                  [self.model.config.label2id[tag]]]
                    while idx + 1 < len(sent) and sent[idx + 1] == f'I-{label}':
                        all_scores.append(
                            probabilities[s_idx][idx+1][self.model.config.label2id[tag]])
                        _, end = offsets[s_idx][idx+1]
                        idx += 1

                    start, end = start.item(), end.item()
                    word = sent_text[s_idx][start:end]
                    score = np.mean(all_scores).item()

                    if label in VERSION_LABELS:
                        word = self._convert_to_version_format(word, label)

                    pred_label.append({
                        "entity_group": label,
                        "word": word,
                        "score": score,
                    })

                idx += 1
        results = self.__combine(pred_label)
        results= self.__remove_duplicate(results)
        return results

    def __combine(self, entities: list):
        results = []
        idx = 0
        entities_size = len(entities)
        vendor = None
        
        while idx < entities_size:
            software = None
            versions = []

            if entities[idx]["entity_group"] == "VENDOR":
                vendor = entities[idx]
                idx += 1
                continue
            if entities[idx]["entity_group"] == "SOFT":
                software = entities[idx]
                while idx+1 < entities_size and entities[idx+1]["entity_group"] in VERSION_LABELS:
                    versions.append(entities[idx+1])
                    idx += 1
            elif entities[idx]["entity_group"] in VERSION_LABELS:
                versions.append(entities[idx])
                while idx + 1 < entities_size and entities[idx+1]["entity_group"] != "SOFT":
                    versions.append(entities[idx+1])
                    idx += 1
            
            idx += 1
            
            results.append(
                {"vendor": vendor, "software": software, "versions": versions})

        if len(results) <= 0 and vendor:
            results.append(
                {"vendor": vendor, "software": None, "versions": []})

        return results

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

        versions = ','.join(versions)

        return versions

    def _convert_to_version_range(self, entity: str):
        one_left = ['start', 'from', '>', '>=']
        one_right = ['prior', 'before', 'through',
                     'to', 'up', 'earlier', '<', '<=','below']

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
                    return self._comfirm_the_boundary(entity, f"{versions[0]},", 1)

            return self._comfirm_the_boundary(entity, f",{versions[0]}", 1)

        if len(versions) == 2:
            return self._comfirm_the_boundary(entity, f"{versions[0]},{versions[1]}", 2)

        if len(versions) >= 3:
            version_str = f"{versions[0]},{versions[-1]}"
            return self._comfirm_the_boundary(entity, version_str, 3)

    def _comfirm_the_boundary(self, entity: str, versions: str, versions_size: int):
        including_key_word = ['include', 'includ', 'through', '=']
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

    def __remove_duplicate_entity(self, entities: list):
        results = []

        idx = 0
        duplicate_index = []
        while idx < len(entities):
            if idx in duplicate_index:
                idx += 1
                continue

            duplicate_index.append(idx)

            current_software = "" if entities[idx].get(
                'software', "") is None else entities[idx].get('software', "").get('word', "")

            current_version = entities[idx].get('versions')

            if current_software is not None:
                ids = idx + 1
                while ids < len(entities) and ids not in duplicate_index:
                    next_software = "" if entities[ids].get(
                        'software', "") is None else entities[ids].get('software', "").get('word', "")
                    if current_software == next_software:
                        for i in entities[ids].get('versions'):
                            current_version.append(i)
                        duplicate_index.append(ids)
                    ids += 1

            results.append({'software': current_software,
                           'versions': current_version})
            idx += 1

        return results

    def __remove_duplicate(self, entities: list):
        results = []

        idx = 0
        duplicate_index = []
        entities_size = len(entities)

        while idx < entities_size:
            if idx in duplicate_index:
                idx += 1
                continue
            duplicate_index.append(idx)

            current_software = "" if entities[idx].get(
                "software", "") is None else entities[idx].get("software", "").get("word", "")
            current_vendor = "" if entities[idx].get(
                "vendor", "") is None else entities[idx].get("vendor", "").get("word", "")
            current_version = entities[idx].get("versions")

            if current_software is not None:
                ids = idx + 1
                while ids < len(entities) and ids not in duplicate_index:
                    next_software = "" if entities[ids].get(
                        "software", "") is None else entities[ids].get("software", "").get("word", "")
                    if current_software == next_software:
                        for i in entities[ids].get("versions"):
                            current_version.append(i)
                        duplicate_index.append(ids)
                    ids += 1

            results.append({"vendor": current_vendor,
                            "software": current_software,
                           "versions": current_version})
            idx += 1

        return results

    def __combine_version(self, entities: list):
        entities_size = len(entities)
        idx = 0
        results = []

        while idx < entities_size:
            software = None
            versions = []

            if entities[idx]['entity_group'] == 'SOFT':
                software = entities[idx]
                while idx+1 < entities_size and entities[idx+1]['entity_group'] in VERSION_LABELS:
                    versions.append(entities[idx+1])
                    idx += 1
            elif entities[idx]['entity_group'] in VERSION_LABELS:
                versions.append(entities[idx])
                while idx + 1 < entities_size and entities[idx+1]['entity_group'] != 'SOFT':
                    versions.append(entities[idx+1])
                    idx += 1

            results.append({'software': software, 'versions': versions})

            idx += 1

        results = self._remove_duplicate_entity(results)

        return results


class BertCrfPipelineBak(Pipeline):
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
        results = self._combine_version(pred_label)
        return results

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
        one_left = ['start', 'from', '>', '>=']
        one_right = ['prior', 'before', 'through',
                     'to', 'up', 'earlier', '<', '<=']

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
        including_key_word = ['include', 'includ', 'through', '=']
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

    def _remove_duplicate_entity(self, entities: list):
        results = []

        idx = 0
        duplicate_index = []
        while idx < len(entities):
            if idx in duplicate_index:
                idx += 1
                continue

            duplicate_index.append(idx)

            current_software = None if entities[idx].get(
                'software', None) is None else entities[idx].get('software', None).get('word', None)

            current_version = entities[idx].get('versions')

            if current_software is not None:
                ids = idx + 1
                while ids < len(entities) and ids not in duplicate_index:
                    next_software = None if entities[ids].get(
                        'software', None) is None else entities[ids].get('software', None).get('word', None)
                    if current_software == next_software:
                        for i in entities[ids].get('versions'):
                            current_version.append(i)
                        duplicate_index.append(ids)
                    ids += 1

            results.append({'software': current_software,
                           'versions': current_version})
            idx += 1

        return results

    def _combine_version(self, entities: list):
        entities_size = len(entities)
        idx = 0
        results = []

        while idx < entities_size:
            software = None
            versions = []

            if entities[idx]['entity_group'] == 'SOFT':
                software = entities[idx]
                while idx+1 < entities_size and entities[idx+1]['entity_group'] != 'SOFT':
                    versions.append(entities[idx+1])
                    idx += 1
            else:
                versions.append(entities[idx])

            results.append({'software': software, 'versions': versions})

            idx += 1

        results = self._remove_duplicate_entity(results)

        return results

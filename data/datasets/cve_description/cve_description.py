import json
import datasets

from datasets import GeneratorBasedBuilder

_DOWNLOAD_URL = './data/datasets/cve_description/cve-500.jsonl'
_DOWNLOAD_TEST_URL = './data/datasets/cve_description/cve-500-test.jsonl'
_DOWNLOAD_TRAIN_URL = './data/datasets/cve_description/cve-500-train.jsonl'


class CveDescriptionConfig(datasets.BuilderConfig):
    # BuilderConfig for CveDescription

    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version('1.0.0'), **kwargs)


class CveDescription(GeneratorBasedBuilder):
    _DESCRIPTION = 'CVE Description Dataset'

    _SV_DESCRIPTION = 'CVE Decription Dataset For Software&Version Recognition'

    BUILDER_CONFIGS = [
        CveDescriptionConfig(
            name='SV',
            description=_SV_DESCRIPTION,
        )
    ]

    def _info(self) -> datasets.DatasetInfo:
        features = {}
        features['id'] = datasets.Value("string")
        features['sentence'] = datasets.Value("string")
        features['labels'] = datasets.features.Sequence({
            "start": datasets.Value("int32"),
            "end": datasets.Value("int32"),
            "label": datasets.Value("string")
        })

        return datasets.DatasetInfo(
            description=self._DESCRIPTION,
            features=datasets.Features(features)
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": _DOWNLOAD_TRAIN_URL
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": _DOWNLOAD_TEST_URL
                }
            )
        ]

    def _generate_examples(self, filepath):

        with open(filepath, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n')):
                if not line:
                    break
                obj = json.loads(line)
                id_ = obj['id']
                sentence = obj['text']
                labels = []

                for label in obj['label']:
                    labels.append(
                        {"start": label[0], "end": label[1], "label": label[2]})

                yield id_, {
                    "id": id_,
                    "sentence": sentence,
                    "labels": labels
                }

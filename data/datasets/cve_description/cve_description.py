import json
import datasets

from datasets import GeneratorBasedBuilder

_DOWNLOAD_URL = 'data\datasets\cve_description\cve-500.jsonl'


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

        information = datasets.DatasetInfo(
            description=self._DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "label": datasets.Sequence(datasets.Value("string"))
                }
            )
        )

        return information

    def _split_generators(self, dl_manager: datasets.DownloadManager):

        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
            "filepath": _DOWNLOAD_URL,
        },),]

    def _generate_examples(self, filepath):

        with open(filepath, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n')):
                if not line:
                    break
                obj = json.loads(line)
                id_ = obj['id']
                sentence = obj['text']
                label = obj['label']
                yield id_, {
                    "id": id_,
                    "sentence": sentence,
                    "label": label
                }

from transformers import Trainer
from datasets import Dataset
from typing import Optional, List, Dict


class NerTrainer(Trainer):
    def evaluate(self, eval_dataset: Optional[Dataset] = None, 
                 ignore_keys: Optional[List[str]] = None, 
                 metric_key_prefix: str = "eval") -> Dict[str, float]:
        
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

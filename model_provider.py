import torch
from config_provider import BertCrfConfig
from torch import nn
from transformers import BertPreTrainedModel, BertModel, BertForTokenClassification
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
from torchcrf import CRF


class BertMlpForNer(BertPreTrainedModel):
    def __init__(self, config) -> None:
        super(BertMlpForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, feature, labels=None):
        outputs = self.bert(**feature)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = logits
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            attention_mask = feature.data['attention_mask']
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                # loss = loss_fct(logits.permute(0, 2, 1), labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            # outputs = (loss,) + outputs
            return loss, outputs
        return outputs


class BertForNer(BertPreTrainedModel):
    def __init__(self, config) -> None:
        super(BertForNer, self).__init__(config)
        self.bert = BertForTokenClassification(config)

    def forward(self, feature, label=None):
        output = self.bert(**feature, labels=label, return_dict=False)
        return output


class BertCrfForNer(BertPreTrainedModel):
    config_class = BertCrfConfig

    def __init__(self, config) -> None:
        super(BertCrfForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, feature, labels=None):
        outputs = self.bert(**feature)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            active_mask = torch.tensor(
                feature.data['attention_mask'], dtype=torch.uint8)
            loss = self.crf(emissions=logits, tags=labels, mask=active_mask)
            outputs = (-1 * loss,) +outputs
        return outputs

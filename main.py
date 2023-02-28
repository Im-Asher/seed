import argparse
import logging
import os
import torch
import torch.cuda
import numpy as np
import json
import time

from tqdm import tqdm
from utils.commom import init_logger, logger
from torch.utils.data import DataLoader
from data.data_utils import collate_fn, load_data, id2label, label2id
from model_provider import BertForNer, BertCrfForNer, BertMlpForNer
from transformers import AutoConfig, AutoTokenizer, AdamW, get_scheduler
from seqeval.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from seqeval.scheme import IOB2
from utils.commom import get_parser
from torch.optim.lr_scheduler import LambdaLR


logger = logging.getLogger(__name__)

MODEL_CLASS = {
    'bert': (AutoConfig, BertForNer, AutoTokenizer),
    'bert-crf': (AutoConfig, BertCrfForNer, AutoTokenizer),
    'bert-mlp': (AutoConfig, BertMlpForNer, AutoTokenizer)
}


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(args: argparse.Namespace, train_dataloader, model: BertForNer, tokenizer, config):

    global_step = 0
    tr_loss = 0.0

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epoch = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epoch

    no_decay = ["bias", "LayerNorm.weight"]

    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
    ]

    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Show Training Parameters
    logger.info("***** Training Parameters Information *****")
    logger.info("Current model name = %s", args.model_type)
    logger.info("Data size = %d", len(train_dataloader)*args.batch_size)
    logger.info("Training epochs = %d", args.num_train_epoch)
    logger.info("Learining rate = %f", args.learning_rate)
    logger.info("Data batch size = %d", args.batch_size)
    logger.info("***** Training Fun *****")

    progress_bar = tqdm(range(len(train_dataloader)), ncols=100)

    for epoch in range(args.num_train_epoch):
        logger.info(
            f'Epoch {epoch +1}/{args.num_train_epoch}\n-------------------')
        progress_bar.reset()
        progress_bar.set_description(f'loss value:{0:>7f}')

        model.train()

        n_epoch_total_loss = 0
        finish_batch_num = epoch * len(train_dataloader)

        for batch, (feature, label) in enumerate(train_dataloader, start=1):
            sentences = feature.pop('sentences')

            feature, label = feature.to(args.device), label.to(args.device)

            loss, _ = model(feature, label)
            tr_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            n_epoch_total_loss += loss.item()

            progress_bar.set_description(
                f'loss value:{n_epoch_total_loss/(finish_batch_num+batch):>7f}')
            progress_bar.update(1)
            global_step += 1

            if args.save_step > 0 and global_step % args.save_step == 0:
                # Save model checkpoint
                output_dir = os.path.join(
                    args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (model.module if hasattr(
                    model, "module") else model)
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                tokenizer.save_vocabulary(output_dir)
                torch.save(scheduler.state_dict(), os.path.join(
                    output_dir, "scheduler.pt"))
        logger.info("\n")
        logger.info(
            f"Epoch {epoch+1}/{args.num_train_epoch}: loss value:{tr_loss/global_step}")

    if 'cuda' in str(args.device):
        torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(config, model: BertForNer, eval_dataloader: DataLoader):
    true_labels, true_predictions = [], []

    model.eval()

    logger.info("start evaluate model...")
    sentences = []
    with torch.no_grad():
        progress_bar = tqdm(range(len(eval_dataloader)))
        for feature, label in eval_dataloader:
            sentences += feature.pop('sentences')
            feature, label = feature.to(args.device), label.to(args.device)
            loss, pred = model(feature, label)
            masks = torch.tensor(feature['attention_mask'], dtype=torch.uint8)
            tags = model.crf.decode(pred, masks)
            predictions = pred[0].argmax(dim=-1).cpu().numpy().tolist()
            labels = label.cpu().numpy().tolist()

            true_labels += [[config.id2label[int(l)]
                             for m, l in zip(mask, label) if m != 0] for mask, label in zip(masks, labels)]

            true_predictions += [
                [config.id2label[int(p)] for (p, l) in zip(
                    prediction, label) if l != -100]
                for prediction, label in zip(tags, labels)
            ]
            progress_bar.update(1)
    fail_sentences = find_failure_sentences(sentences,true_labels,true_predictions)
    logger.info(classification_report(true_labels,
                                      true_predictions, mode='strict', scheme=IOB2))
    f1_value = f1_score(true_labels, true_predictions)
    precision_value = precision_score(true_labels, true_predictions)
    recall_value = recall_score(true_labels, true_predictions)
    accuracy_value = accuracy_score(true_labels, true_predictions)
    print(f'f1_value:{f1_value},precision_value:{precision_value},recall_value:{recall_value},accuracy_value:{accuracy_value}')
    return f1_value


def predict(args: argparse.Namespace, model: BertForNer, predict_dataloader: DataLoader, tokenizer: AutoTokenizer):
    logger.info("start predict ...")
    progress_bar = tqdm(range(len(predict_dataloader)))
    results = []

    for idx in range(len(predict_dataloader)):
        example = predict_dataloader[idx]
        inputs = tokenizer(example['sentence'],
                           truncation=True, return_tensors='pt')
        inputs = inputs.to(args.device)
        pred = model(inputs)

        probabilities = torch.nn.functional.softmax(
            pred, dim=-1)[0].cpu().numpy().tolist()
        predictions = pred.argmax(dim=-1)[0].cpu().numpy().tolist()

        pred_label = []
        inputs_with_offsets = tokenizer(
            example['sentence'], return_offsets_mapping=True)
        tokens = inputs_with_offsets.tokens()
        offsets = inputs_with_offsets["offset_mapping"]

        idx = 0
        while idx < len(predictions):
            pred = predictions[idx]
            label = id2label[pred]
            if label != "O":
                label = label[2:]  # Remove the B- or I-
                start, end = offsets[idx]
                all_scores = [probabilities[idx][pred]]
                # Grab all the tokens labeled with I-label
                while (
                    idx + 1 < len(predictions) and
                    id2label[predictions[idx + 1]] == f"I-{label}"
                ):
                    all_scores.append(
                        probabilities[idx + 1][predictions[idx + 1]])
                    _, end = offsets[idx + 1]
                    idx += 1

                score = np.mean(all_scores).item()
                word = example['sentence'][start:end]
                pred_label.append(
                    {
                        "entity_group": label,
                        "score": score,
                        "word": word,
                        "start": start,
                        "end": end,
                    }
                )
            idx += 1
        results.append(
            {
                "sentence": example['sentence'],
                "pred_label": pred_label,
                "true_label": example['labels']
            }
        )
    with open(args.predict_result_dir, 'wt', encoding='utf-8') as f:
        for exapmle_result in results:
            f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')


def find_failure_sentences(sentences: list, true_labels: list, true_predictions: list):
    fail_sentences = []
    idx = 0
    for l, p in zip(true_labels, true_predictions):
        if l != p:
            fail_sentences.append(sentences[idx])
        idx += 1
    with open ('failure_sentences.txt','a+',encoding='utf-8') as f:
        for line in fail_sentences:
            f.write(line)
            f.write('\n')
    return fail_sentences


if __name__ == "__main__":

    args = get_parser().parse_args()
    # Initial output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    log_dir = args.output_dir + '/log'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # Initial logger
    time_ = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    init_logger(log_file=log_dir +
                f'/{args.model_type}-{args.task_name}-{time_}.log')

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_class, model_class, tokenizer_class = MODEL_CLASS[args.model_type]

    config = config_class.from_pretrained(args.name_or_path)
    config.id2label = id2label
    config.label2id = label2id
    config.loss_type = args.loss_type

    model = model_class.from_pretrained(
        args.name_or_path, config=config).to(args.device)

    tokenizer = tokenizer_class.from_pretrained(
        args.name_or_path, do_lower_case=True)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataloader = DataLoader(load_data(args.train_file)[
            :400], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        train(args=args, train_dataloader=train_dataloader,
              model=model, tokenizer=tokenizer, config=config)

        # Save trained model/tokenizer/training parameters
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to {}".format(args.output_dir))

        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=True)
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        eval_dataloader = DataLoader(load_data(args.train_file)[
                                     400:], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        for checkpoint in checkpoints:
            config = config_class.from_pretrained(checkpoint)
            model = model_class.from_pretrained(
                checkpoint, config=config).to(args.device)
            result = evaluate(config=config, model=model,
                              eval_dataloader=eval_dataloader)
            results.update({checkpoint: result})
        output_eval_results = os.path.join(args.output_dir, "eval_resutls.txt")

        with open(output_eval_results, 'w') as f:
            for key in sorted(results.keys()):
                f.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict:
        pass

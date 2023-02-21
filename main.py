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
from seqeval.metrics import classification_report,f1_score,accuracy_score,precision_score,recall_score
from seqeval.scheme import IOB2
from utils.commom import get_parser


logger = logging.getLogger(__name__)

MODEL_CLASS = {
    'bert': (AutoConfig, BertForNer, AutoTokenizer),
    'bert-crf': (AutoConfig, BertCrfForNer, AutoTokenizer),
    'bert-mlp': (AutoConfig, BertMlpForNer, AutoTokenizer)
}


def train(args: argparse.Namespace, train_dataloader, model: BertForNer, tokenizer, config):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler('linear', optimizer=optimizer,
                                 num_warmup_steps=0, num_training_steps=args.num_training_steps)
    global_step = 0
    tr_loss = 0.0

    # Show Training Parameters
    logger.info("***** Training Parameters Information *****")
    logger.info("Data size = %d", len(train_dataloader)*args.batch_size)
    logger.info("Training epochs = %d", args.epoch)
    logger.info("Learining rate = %f", args.learning_rate)
    logger.info("Data batch size = %d", args.batch_size)
    logger.info("***** Training Fun *****")

    progress_bar = tqdm(range(len(train_dataloader)), ncols=100)

    for epoch in range(args.epoch):
        logger.info(f'Epoch {epoch +1}/{args.epoch}\n-------------------')
        progress_bar.reset()
        progress_bar.set_description(f'loss value:{0:>7f}')

        model.train()

        n_epoch_total_loss = 0
        finish_batch_num = epoch * len(train_dataloader)

        for batch, (feature, label) in enumerate(train_dataloader, start=1):
            feature, label = feature.to(args.device), label.to(args.device)

            loss, logits = model(feature, label)
            tr_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

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
                torch.save(lr_scheduler.state_dict(), os.path.join(
                    output_dir, "lr_scheduler.pt"))
        logger.info("\n")
        logger.info(
            f"Epoch {epoch+1}/{args.epoch}: loss value:{tr_loss/global_step}")

    if 'cuda' in str(args.device):
        torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(config, model: BertForNer, eval_dataloader: DataLoader):
    true_labels, true_predictions = [], []

    model.eval()

    logger.info("start evaluate model...")

    with torch.no_grad():
        progress_bar = tqdm(range(len(eval_dataloader)))
        for feature, label in eval_dataloader:
            pred = model(feature)
            predictions = pred[0].argmax(dim=-1).cpu().numpy().tolist()
            labels = label.cpu().numpy().tolist()
            true_labels += [[config.id2label[int(l)]
                             for l in label if l !=-100] for label in labels]
            true_predictions += [
                [config.id2label[int(p)] for (p, l) in zip(
                    prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            progress_bar.update(1)
    logger.info(classification_report(true_labels,
          true_predictions, mode='strict', scheme=IOB2))
    f1_value = f1_score(true_labels,true_predictions)
    precision_value = precision_score(true_labels,true_predictions)
    recall_value = recall_score(true_labels,true_predictions)
    accuracy_value = accuracy_score(true_labels,true_predictions)
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

    config = config_class.from_pretrained(args.check_point)
    config.id2label = id2label
    config.label2id = label2id
    config.loss_type = args.loss_type

    model = model_class.from_pretrained(
        args.check_point, config=config).to(args.device)

    tokenizer = tokenizer_class.from_pretrained(
        args.check_point, do_lower_case=True)

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
            results.update({checkpoint:result})
        output_eval_results = os.path.join(args.output_dir, "eval_resutls.txt")

        with open(output_eval_results, 'w') as f:
            for key in sorted(results.keys()):
                f.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict:
        pass

import argparse
import logging
import torch
import torch.cuda
import numpy as np
import json

from tqdm import tqdm
from torch.utils.data import DataLoader
from data.data_utils import collate_fn, load_data, id2label, label2id
from model_provider import BertForNer
from transformers import AutoConfig, AutoTokenizer, AdamW, get_scheduler
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

DATA_DIR = './data/dataset/'
TRAIN_FILE = DATA_DIR + 'cve-500.jsonl'
EVAL_FILE = DATA_DIR + 'cve-500.jsonl'
PREDICT_FILE = DATA_DIR + 'cve-500.jsonl'


CHECK_POINT = 'bert-base-uncased'

OUTPUR_DIR = './model_cache/'
PREDICT_RESULT_DIR = OUTPUR_DIR+'predict_result.json'

logger = logging.getLogger(__name__)


def train(args: argparse.Namespace, train_dataloader, model: BertForNer, tokenizer, config):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler('linear', optimizer=optimizer,
                                 num_warmup_steps=0, num_training_steps=args.num_training_steps)

    # Show Training Parameters
    logger.info("***** Training Parameters Information *****")
    logger.info("Data size = %d", len(train_dataloader))
    logger.info("Training epochs = %d", args.epoch)
    logger.info("Learining rate = %f", args.learning_rate)
    logger.info("Data batch size = %d", args.batch_size)
    logger.info("***** Training Fun *****")

    for epoch in range(args.epoch):
        logger.info(f'Epoch {epoch +1}/{args.epoch}\n-------------------')
        progress_bar = tqdm(range(len(train_dataloader)))
        progress_bar.set_description(f'loss value:{0:>7f}')

        model.train()

        n_epoch_total_loss = 0
        finish_batch_num = epoch * len(train_dataloader)

        for batch, (feature, label) in enumerate(train_dataloader, start=1):
            feature, label = feature.to(args.device), label.to(args.device)

            loss, _ = model(feature, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            n_epoch_total_loss += loss.item()

            progress_bar.set_description(
                f'loss value:{n_epoch_total_loss/(finish_batch_num+batch):>7f}')
            progress_bar.update(1)


def evaluate(config, model: BertForNer, eval_dataloader: DataLoader):
    true_labels, true_predictions = [], []

    model.eval()

    logger.info("start evaluate model...")

    with torch.no_grad():
        progress_bar = tqdm(range(len(eval_dataloader)))
        for feature, label in eval_dataloader:
            pred = model(feature)
            predictions = pred.argmax(dim=-1).cpu().numpy().tolist()
            labels = label.cpu().numpy().tolist()
            true_labels += [[config.id2label[int(l)]
                             for l in label if l != 0] for label in labels]
            true_predictions += [
                [config.id2label[int(p)] for (p, l) in zip(
                    prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            progress_bar.update(1)
    print(classification_report(true_labels,
          true_predictions, mode='strict', scheme=IOB2))


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
    parser = argparse.ArgumentParser('NLP model parameter setting.')

    # Required parameters
    parser.add_argument('--data_dir', default=DATA_DIR,
                        type=str, help='The input data dir.')
    parser.add_argument('--check_point', default=CHECK_POINT, type=str,
                        help='The name of pre-traind model name or path in local')
    parser.add_argument('--train_file', default=TRAIN_FILE,
                        type=str, help='train file for model training')
    parser.add_argument('--evaluate_file', default=EVAL_FILE,
                        type=str, help='evaluate file for model evaluate')
    parser.add_argument('--predict_file', default=PREDICT_FILE,
                        type=str, help='predict file for model predict')
    parser.add_argument('--output_dir', default=OUTPUR_DIR, type=str,
                        help='The output directory where the model trained will be written ')
    parser.add_argument('--predict_result_dir', default=PREDICT_RESULT_DIR, type=str,
                        help='The predict result file path when do predict')

    # Optional parameters
    # train parameters
    parser.add_argument('--batch_size', default=10,
                        type=int, help='The dataset batch size')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Total number of training epochs to perform')
    parser.add_argument('--max_train_step', default=8600,
                        type=int, help='Max step for model training')
    parser.add_argument('--learning_rate', default=2e-5, type=float,
                        help='Default learning rate for model training.')
    parser.add_argument('--num_training_steps', default=8600, type=int,
                        help='training step.')
    parser.add_argument('--loss_type', default='ce', type=str,
                        help="loss function type ('lsr', 'focal', 'ce')")
    # runing mode
    parser.add_argument('--do_train', default=True, action='store_true',
                        help='Whether to run train process')
    parser.add_argument('--do_eval', default=False, action='store_true',
                        help='Whether to run evaluate process')
    parser.add_argument('--do_predict', default=False, action='store_true',
                        help='Whether to run predict process')

    # model/config/tokenizer setting
    parser.add_argument('--config_name', default=CHECK_POINT,
                        type=str, help='Pretrained config check point')
    parser.add_argument('--model_name', default=CHECK_POINT,
                        type=str, help='Pretrained model check point')
    parser.add_argument('--tokenizer_name', default=CHECK_POINT,
                        type=str, help='Pretrained tokenizer check point')

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = AutoConfig.from_pretrained(args.check_point)
    config.id2label = id2label
    config.label2id = label2id
    config.loss_type = args.loss_type
    model = BertForNer.from_pretrained(
        args.check_point, config=config).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.check_point)

    if args.do_train:
        train_dataloader = DataLoader(load_data(args.train_file)[
            :80], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        train(args=args, train_dataloader=train_dataloader,
              model=model, tokenizer=tokenizer, config=config)

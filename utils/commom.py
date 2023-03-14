import logging
import argparse

from pathlib import Path

TASK_NAME = 'SV'
DATA_DIR = './data/datasets/'
TRAIN_FILE = DATA_DIR + TASK_NAME + '/cve-1000.jsonl'
EVAL_FILE = DATA_DIR + TASK_NAME + '/cve-1000.jsonl'
PREDICT_FILE = DATA_DIR + TASK_NAME + '/cve-1000.jsonl'

CHECK_POINT = 'bert-base-uncased'

OUTPUT_DIR = './model_cache/'
PREDICT_RESULT_DIR = OUTPUT_DIR+'predict_result.json'

logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def get_parser():
    parser = argparse.ArgumentParser('NLP model parameter setting.')

    # Required parameters
    parser.add_argument('--data_dir', default=DATA_DIR+TASK_NAME,
                        type=str, help='The input data dir.')
    parser.add_argument('--task_name', default=TASK_NAME,
                        type=str,  help='Execute task name using model')
    parser.add_argument('--name_or_path', default=CHECK_POINT, type=str,
                        help='The name of pre-traind model name or path in local')
    parser.add_argument('--train_file', default=TRAIN_FILE,
                        type=str,  help='Train file for model training')
    parser.add_argument('--evaluate_file', default=EVAL_FILE,
                        type=str,  help='Evaluate file for model evaluate')
    parser.add_argument('--predict_file', default=PREDICT_FILE,
                        type=str,  help='Predict file for model predict')
    parser.add_argument('--model_type', default="bert-crf",
                        type=str,  help="Please select model type!")
    parser.add_argument('--output_dir', default=OUTPUT_DIR, type=str,
                        help='The output directory where the model trained will be written ')
    parser.add_argument('--predict_result_dir', default=PREDICT_RESULT_DIR, type=str,
                        help='The predict result file path when do predict')

    # Optional parameters
    # train parameters
    parser.add_argument('--batch_size', default=4,
                        type=int, help='The dataset batch size')
    parser.add_argument('--num_train_epoch', default=3, type=int,
                        help='Total number of training epochs to perform')
    parser.add_argument('--max_train_step', default=8600,
                        type=int, help='Max step for model training')
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='Default learning rate for model training.')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")  
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument('--crf_learning_rate', default=1e-3, type=float,
                        help='Default learning rate for model training.')
    parser.add_argument('--num_training_steps', default=8600, type=int,
                        help='Training step.')
    parser.add_argument('--logging_step', default=1000, type=int,
                        help="Log checkpoint every X updates steps")
    parser.add_argument('--save_step', default=-1, type=int,
                        help="Save checkpoint every X updates steps")
    parser.add_argument('--eval_step', default=1000, type=int,
                        help="Save checkpoint every X updates steps")
    parser.add_argument('--loss_type', default='ce', type=str,
                        help="loss function type ('lsr', 'focal', 'ce')")
    parser.add_argument('--warmup_proportion', default=0.1,
                        type=float, help="Set warm up proportion")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument("--reduction", default="token_mean", type=str,
                        help="CRF reduction ['sum','mean','token_mean']", )
            
    # runing mode
    parser.add_argument('--do_train', default=True, action='store_true',
                        help='Whether to run train process')
    parser.add_argument('--do_eval', default=False, action='store_true',
                        help='Whether to run evaluate process')
    parser.add_argument('--do_predict', default=False, action='store_true',
                        help='Whether to run predict process')

    # tokenizer setting
    parser.add_argument('--do_lower_case', default=True,
                        action='store_true', help='Pretrained config check point')

    return parser

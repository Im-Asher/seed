import logging
import argparse

from pathlib import Path

DATA_DIR = './data/dataset/'
TRAIN_FILE = DATA_DIR + 'cve-500.jsonl'
EVAL_FILE = DATA_DIR + 'cve-500.jsonl'
PREDICT_FILE = DATA_DIR + 'cve-500.jsonl'


CHECK_POINT = 'bert-base-uncased'

OUTPUR_DIR = './model_cache/'
PREDICT_RESULT_DIR = OUTPUR_DIR+'predict_result.json'

logger = logging.getLogger()

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file,Path):
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
    parser.add_argument('--data_dir', default=DATA_DIR,
                        type=str, help='The input data dir.')
    parser.add_argument('--task_name', default="SV",
                        type=str,  help='Execute task name using model')
    parser.add_argument('--check_point', default=CHECK_POINT, type=str,
                        help='The name of pre-traind model name or path in local')
    parser.add_argument('--train_file', default=TRAIN_FILE,
                        type=str,  help='Train file for model training')
    parser.add_argument('--evaluate_file', default=EVAL_FILE,
                        type=str,  help='Evaluate file for model evaluate')
    parser.add_argument('--predict_file', default=PREDICT_FILE,
                        type=str,  help='Predict file for model predict')
    parser.add_argument('--model_type', default="BERT",
                        type=str,  help="Model type(ONLY BERT!!! NOW)")
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
                        help='Training step.')
    parser.add_argument('--save_step', default=1000, type=int,
                        help="Save checkpoint every X updates steps")
    parser.add_argument('--loss_type', default='ce', type=str,
                        help="loss function type ('lsr', 'focal', 'ce')")
    # runing mode
    parser.add_argument('--do_train', default=False, action='store_true',
                        help='Whether to run train process')
    parser.add_argument('--do_eval', default=True, action='store_true',
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

    return parser
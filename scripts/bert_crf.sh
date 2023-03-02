CURRENT_DIR=`pwd`
TASK_NAME="SV"

export BERT_BASE_DIR=$CURRENT_DIR/model_cache/bert-base-uncased
export DATA_DIR=$CURRENT_DIR/data/datasets
export OUTPUT_DIR=$CURRENT_DIR/model_cache/


python3 main.py \
    --model_type=bert-crf \
    --name_or_path=bert-base-uncased \
    --task_name=$TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir=$DATA_DIR/${TASK_NAME}/ \
    --learning_rate=5e-5 \
    --crf_learning_rate=1e-3 \
    --num_train_epoch=3 \
    --logging_step=-1 \
    --save_step=-1 \
    --output_dir=$OUTPUT_DIR
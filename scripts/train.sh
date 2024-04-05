# ACOS-Laptop train
CUDA_VISIBLE_DEVICES=1 nohup python run.py \
  --train_batch_size 4 \
  --data_path ./data/ACOS/v2/laptop/ \
  --task ACOS \
  --data_type laptop \
  --model_path ../pretrained-models/SentiWSP \
  --do_train \
  --learning_rate1 1e-4 \
  --learning_rate2 1e-5 \
  --use_category_SCL \
  --use_sentiment_SCL \
  > output.txt 2>&1 &
# ACOS-Rest train
CUDA_VISIBLE_DEVICES=1 nohup python run.py \
  --train_batch_size 4 \
  --data_path ./data/ACOS/v2/rest/ \
  --task ACOS \
  --data_type rest \
  --model_path ../pretrained-models/SentiWSP \
  --do_train \
  --learning_rate1 1e-4 \
  --learning_rate2 1e-5 \
  --use_category_SCL \
  --use_sentiment_SCL \
  > output.txt 2>&1 &
# ===================================================================================
# ASQP-Rest15 train
python run.py \
  --train_batch_size 4 \
  --data_path ../data/ASQP/v2/rest15/ \
  --task ASQP \
  --data_type rest15 \
  --model_path ../pretrained-models/SentiWSP \
  --do_train \
  --learning_rate1 3e-5 \
  --learning_rate2 1e-5 \
  --use_category_SCL \
  --use_sentiment_SCL \
  > output.txt 2>&1 &
# ASQP-Rest16 train
python run.py \
  --train_batch_size 4 \
  --data_path ../data/ASQP/v2/rest16/ \
  --task ASQP \
  --data_type rest16 \
  --model_path ../pretrained-models/SentiWSP \
  --do_train \
  --learning_rate1 3e-4 \
  --learning_rate2 3e-5 \
  --use_category_SCL \
  --use_sentiment_SCL \
  > output.txt 2>&1 &
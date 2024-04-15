# ASQP-Rest15 train
python run.py \
  --train_batch_size 4 \
  --data_path ./data/ASQP/rest15/v2/ \
  --task ASQP \
  --data_type rest15 \
  --model_path ../pretrained-models/SentiWSP \
  --learning_rate1 3e-5 \
  --learning_rate2 1e-5 \
  --use_category_SCL \
  --use_sentiment_SCL \
  --contrastive_lr1 1e-4 \
  --contrastive_lr2 1e-5 \
  --do_train

# ASQP-Rest16 train
python run.py \
  --train_batch_size 4 \
  --data_path ./data/ASQP/rest16/v2/ \
  --task ASQP \
  --data_type rest16 \
  --model_path ../pretrained-models/SentiWSP \
  --learning_rate1 3e-4 \
  --learning_rate2 3e-5 \
  --use_category_SCL \
  --use_sentiment_SCL \
  --contrastive_lr1 3e-4 \
  --contrastive_lr2 3e-5 \
  --do_train

# ===================================================================================
# ACOS-Rest train
python run.py \
  --train_batch_size 4 \
  --data_path ./data/ACOS/v2/rest/ \
  --task ACOS \
  --data_type rest \
  --model_path ../pretrained-models/SentiWSP \
  --learning_rate1 3e-5 \
  --learning_rate2 1e-5 \
  --use_category_SCL \
  --use_sentiment_SCL \
  --contrastive_lr1 3e-5 \
  --contrastive_lr2 1e-5 \
  --do_train

# ACOS-Laptop train
python run.py \
  --train_batch_size 4 \
  --data_path ./data/ACOS/v2/laptop/ \
  --task ACOS \
  --data_type laptop \
  --model_path ../pretrained-models/SentiWSP \
  --learning_rate1 3e-5 \
  --learning_rate2 1e-5 \
  --use_category_SCL \
  --use_sentiment_SCL \
  --contrastive_lr1 3e-5 \
  --contrastive_lr2 1e-5 \
  --do_train
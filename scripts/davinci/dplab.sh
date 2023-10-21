#!/usr/bin/env bash
# Params:
# $1: train, val, test, segfix
# $2: 1, 2, 3... index of run
# $3: if segfix, val or test
# $4: ss, ms
P_PATH="../.."
P_NAME="davinci"
DATA_DIR="/tmp/512"
SAVE_DIR="${P_PATH}/output/${P_NAME}"
BACKBONE="deepbase_resnet101_dilated8"

CONFIGS="${P_PATH}/configs/${P_NAME}/R_101_D_8.json"
CONFIGS_TEST="${P_PATH}/configs/davinci/R_101_D_8_TEST.json"

MODEL_NAME="deeplab_v3"
LOSS_TYPE="fs_auxce_loss"
CHECKPOINTS_ROOT="checkpoints/${P_NAME}"
CHECKPOINTS_NAME="${MODEL_NAME}_"$2
LOG_FILE="${P_PATH}/log/${P_NAME}/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"

PRETRAINED_MODEL="${P_PATH}/pretrained_model/resnet101-5d3b4d8f.pth"
MAX_ITERS=90000
BATCH_SIZE=8
BASE_LR=0.008

if [ "$1"x == "train"x ]; then
  python ${P_PATH}/main.py --configs ${CONFIGS} \
    --drop_last y \
    --phase train \
    --gathered y \
    --loss_balance y \
    --log_to_file n \
    --backbone ${BACKBONE} \
    --model_name ${MODEL_NAME} \
    --gpu 1 3 \
    --data_dir ${DATA_DIR} \
    --loss_type ${LOSS_TYPE} \
    --max_iters ${MAX_ITERS} \
    --checkpoints_root ${CHECKPOINTS_ROOT} \
    --checkpoints_name ${CHECKPOINTS_NAME} \
    --pretrained ${PRETRAINED_MODEL} \
    --train_batch_size ${BATCH_SIZE} \
    --base_lr ${BASE_LR} \
    --distributed \
    2>&1 | tee ${LOG_FILE}


elif [ "$1"x == "resume"x ]; then
  python -u ${P_PATH}/main.py --configs ${CONFIGS} \
    --drop_last y \
    --phase train \
    --gathered y \
    --loss_balance y \
    --log_to_file n \
    --backbone ${BACKBONE} \
    --model_name ${MODEL_NAME} \
    --max_iters ${MAX_ITERS} \
    --data_dir ${DATA_DIR} \
    --loss_type ${LOSS_TYPE} \
    --gpu 1 3 \
    --checkpoints_name ${CHECKPOINTS_NAME} \
    --resume_continue y \
    --resume ${P_PATH}/checkpoints/davinci/${CHECKPOINTS_NAME}_max_performance.pth \
    --train_batch_size ${BATCH_SIZE} \
    --distributed \
    2>&1 | tee -a ${LOG_FILE}


elif [ "$1"x == "val"x ]; then
  python -u ${P_PATH}/main.py \
    --configs ${CONFIGS} \
    --drop_last y \
    --phase val \
    --gathered y \
    --loss_balance y \
    --log_to_file n \
    --backbone ${BACKBONE} \
    --model_name ${MODEL_NAME} \
    --max_iters ${MAX_ITERS} \
    --data_dir ${DATA_DIR} \
    --loss_type ${LOSS_TYPE} \
    --gpu 1 3 \
    --checkpoints_name ${CHECKPOINTS_NAME} \
    --resume_continue y \
    --resume ${P_PATH}/checkpoints/davinci/${CHECKPOINTS_NAME}_max_performance.pth \
    --train_batch_size ${BATCH_SIZE} \
    --distributed \
    2>&1 | tee -a ${LOG_FILE}


  #   --configs ${CONFIGS} \
  #   --drop_last y  \
  #   --data_dir ${DATA_DIR} \
  #   --backbone ${BACKBONE} \
  #   --model_name ${MODEL_NAME} \
  #   --checkpoints_name ${CHECKPOINTS_NAME} \
  #   --phase test \
  #   --gpu 0 2 \
  #   --resume ${P_PATH}/checkpoints/davinci/${CHECKPOINTS_NAME}_max_performance.pth \
  #   --loss_type ${LOSS_TYPE} \
  #   --test_dir ${DATA_DIR}/test \
  #   --out_dir ${SAVE_DIR}/${CHECKPOINTS_NAME}_val 

  # python -u ${P_PATH}/lib/metrics/cityscapes_evaluator.py \
  #   --pred_dir ${SAVE_DIR}/${CHECKPOINTS_NAME}_val_ms/label  \
  #   --gt_dir ${DATA_DIR}/val/label



elif [ "$1"x == "test"x ]; then
  python -u ${P_PATH}/main.py \
    --configs ${CONFIGS} \
    --drop_last y \
    --data_dir ${DATA_DIR} \
    --backbone ${BACKBONE} \
    --model_name ${MODEL_NAME} \
    --checkpoints_name ${CHECKPOINTS_NAME} \
    --phase test \
    --gpu 0 2 \
    --resume ${P_PATH}/checkpoints/davinci/${CHECKPOINTS_NAME}_max_performance.pth \
    --test_dir ${DATA_DIR}/test \
    --out_dir ${SAVE_DIR}/${CHECKPOINTS_NAME}_t 
else
  echo "$1"x" is invalid..."
fi

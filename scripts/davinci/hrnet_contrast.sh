#!/usr/bin/env bash
# Params:
# $1: train, val, test, segfix
# $2: 1, 2, 3... index of run

P_PATH="../.."
P_NAME="davinci"
DATA_DIR="/tmp/512"
SAVE_DIR="${P_PATH}/output/${P_NAME}"
BACKBONE="hrnet48"

CONFIGS="${P_PATH}/configs/${P_NAME}/H_48_D_4.json"
CONFIGS_TEST="${P_PATH}/configs/cdavinci/H_48_D_4_TEST.json"

MODEL_NAME="hrnet_w48_contrast"
LOSS_TYPE="contrast_ce_loss"
CHECKPOINTS_ROOT="checkpoints/${P_NAME}/"
CHECKPOINTS_NAME="${MODEL_NAME}_"$2
LOG_FILE="${P_PATH}/log/${P_NAME}/${CHECKPOINTS_NAME}"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="${P_PATH}/pretrained_model/hrnetv2_w48_imagenet_pretrained.pth"
MAX_ITERS=89400
BATCH_SIZE=8
BASE_LR=0.01

if [ "$1"x == "train"x ]; then
  python ${P_PATH}/main_contrastive.py \
    --configs ${CONFIGS} \
    --drop_last y \
    --phase train \
    --gathered y \
    --loss_balance y \
    --pretrained ${PRETRAINED_MODEL} \
    --log_to_file n \
    --log_file ${LOG_FILE} \
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
  python -u ${P_PATH}/main_contrastive.py \
    --configs ${CONFIGS} \
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
    --checkpoints_root ${CHECKPOINTS_ROOT} \
    --checkpoints_name ${CHECKPOINTS_NAME} \
    --resume_continue y \
    --resume /home/duhj/ContrastiveSeg/checkpoints/davinci/hrnet_w48_contrast_9-ce*0.5_latest.pth \
    --train_batch_size ${BATCH_SIZE} \
    --distributed \
    2>&1 | tee -a ${LOG_FILE}

elif [ "$1"x == "val"x ]; then
  python ${P_PATH}/main_contrastive.py \
    --configs ${CONFIGS} \
    --drop_last y \
    --data_dir ${DATA_DIR} \
    --backbone ${BACKBONE} \
    --model_name ${MODEL_NAME} \
    --checkpoints_name ${CHECKPOINTS_NAME} \
    --phase val \
    --gpu 1 3 \
    --resume ${P_PATH}/checkpoints/davinci/${CHECKPOINTS_NAME}_max_performance.pth \
    --loss_type ${LOSS_TYPE} \
    --test_dir ${DATA_DIR}/val/image \
    --out_dir ${SAVE_DIR}/${CHECKPOINTS_NAME}_val \
    --distributed

elif [ "$1"x == "test"x ]; then
  python ${P_PATH}/main_contrastive.py \
    --configs ${CONFIGS} \
    --drop_last y \
    --data_dir ${DATA_DIR} \
    --backbone ${BACKBONE} \
    --model_name ${MODEL_NAME} \
    --checkpoints_name ${CHECKPOINTS_NAME} \
    --phase test \
    --gpu 1 3 \
    --resume ${P_PATH}/checkpoints/davinci/${CHECKPOINTS_NAME}_max_performance.pth \
    --loss_type ${LOSS_TYPE} \
    --test_dir ${DATA_DIR}/test/image \
    --log_to_file n \
    --out_dir ${SAVE_DIR}/${CHECKPOINTS_NAME}_test \
    --distributed
elif [ "$1" == "tsne" ]; then
  python -u ${P_PATH}/main_contrastive.py \
    --configs ${CONFIGS} \
    --drop_last y \
    --phase tsne \
    --gathered y \
    --loss_balance y \
    --log_to_file n \
    --backbone ${BACKBONE} \
    --model_name ${MODEL_NAME} \
    --max_iters ${MAX_ITERS} \
    --data_dir ${DATA_DIR} \
    --loss_type ${LOSS_TYPE} \
    --gpu 3 \
    --checkpoints_name ${CHECKPOINTS_NAME} \
    --resume_continue y \
    --resume ${P_PATH}/checkpoints/davinci/${CHECKPOINTS_NAME}_max_performance.pth \
    --train_batch_size ${BATCH_SIZE} \
    --distributed \
    2>&1 | tee -a ${LOG_FILE}_tsne


else
  echo "$1"x" is invalid..."
fi

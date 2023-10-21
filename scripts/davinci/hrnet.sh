#!/usr/bin/env bash
# Params:
# $1: train, val, test, segfix
# $2: 1, 2, 3... index of run
# $3: if segfix, val or test
# $4: ss, ms
P_PATH="../.."
P_NAME="davinci"
DATA_DIR="/tmp/512"
SAVE_DIR="${P_PATH}/output/${P_NAME}_1"
BACKBONE="hrnet48"

CONFIGS="${P_PATH}/configs/${P_NAME}/H_48_D_4.json"
CONFIGS_TEST="${P_PATH}/configs/cdavinci/H_48_D_4_TEST.json"

MODEL_NAME="hrnet_w48"
LOSS_TYPE="fs_ce_loss"
CHECKPOINTS_ROOT="checkpoints/${P_NAME}"
CHECKPOINTS_NAME="${MODEL_NAME}_"$2
LOG_FILE="${P_PATH}/log/${P_NAME}/${CHECKPOINTS_NAME}"
echo "Logging to $LOG_FILE"

PRETRAINED_MODEL="${P_PATH}/pretrained_model/hrnetv2_w48_imagenet_pretrained.pth"
MAX_ITERS=89400
BATCH_SIZE=8
BASE_LR=0.01

if [ "$1"x == "train"x ]; then
  python ${P_PATH}/main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered y \
                       --loss_balance y \
                       --log_to_file n \
                       --log_file ${LOG_FILE} \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 0 1 \
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
  python -u ${P_PATH}/main.py \
                       --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered y \
                       --loss_balance y \
                       --log_to_file y \
                       --log_file ${LOG_FILE}_2 \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --max_iters 70000 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --gpu 1 3 \
                       --checkpoints_root ${CHECKPOINTS_ROOT} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --resume_continue y \
                       --resume ${P_PATH}/${CHECKPOINTS_ROOT}/${CHECKPOINTS_NAME}_max_performance.pth \
                       --train_batch_size ${BATCH_SIZE} \
                       --distributed \
                        # 2>&1 | tee -a ${LOG_FILE}


elif [ "$1"x == "val"x ]; then
  python -u ${P_PATH}/main.py --configs ${CONFIGS} --drop_last y  --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0 1 2 3 --resume ${CHECKPOINTS_ROOT}/checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --loss_type ${LOSS_TYPE} --test_dir ${DATA_DIR}/val/image \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms

  python -m lib.metrics.cityscapes_evaluator --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms/label  \
                                       --gt_dir ${DATA_DIR}/val/label

elif [ "$1"x == "segfix"x ]; then
  if [ "$3"x == "test"x ]; then
    DIR=${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss/label
    echo "Applying SegFix for $DIR"
    ${PYTHON} scripts/cityscapes/segfix.py \
      --input $DIR \
      --split test \
      --offset ${DATA_ROOT}/cityscapes/test_offset/semantic/offset_hrnext/
  elif [ "$3"x == "val"x ]; then
    DIR=${SAVE_DIR}${CHECKPOINTS_NAME}_val/label
    echo "Applying SegFix for $DIR"
    ${PYTHON} scripts/cityscapes/segfix.py \
      --input $DIR \
      --split val \
      --offset ${DATA_ROOT}/cityscapes/val/offset_pred/semantic/offset_hrnext/
  fi

elif [ "$1"x == "test"x ]; then
  if [ "$5"x == "ss"x ]; then
    echo "[single scale] test"
    python -u main.py --configs ${CONFIGS} --drop_last y --data_dir ${DATA_DIR} \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ${CHECKPOINTS_ROOT}/checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss
  else
    echo "[multiple scale + flip] test"
    python -u ${P_PATH}/main.py --configs ${CONFIGS_TEST} --drop_last y --data_dir ${DATA_DIR} \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ${CHECKPOINTS_ROOT}/checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ms
  fi

else
  echo "$1"x" is invalid..."
fi
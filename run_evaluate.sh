#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0  


#!/bin/bash

WEIGHT_PATH="/home/work/SegTrimap/output/loss/all_dataset_alphamatte_vit_huge448_BCE_loss/000/checkpoints/last_checkpoint.pth"
ALPHA_DATASET_PATH="./datasets/AlphaDataset/Adobe_Composition_test"
LOG_DIR="./evaluation/alphamatte/eval_logs"

python evaluation/alphamatte/eval_alpha.py \
    --weight_path "$WEIGHT_PATH" \
    --device "cpu" \
    --batch_size 20 \
    --log_dir "$LOG_DIR" \
    --alpha_dataset_path "$ALPHA_DATASET_PATH"



#kjk20250508
# WEIGHT_PATH="/home/work/SegTrimap/output/loss/all_dataset_trimap_vit_huge448_CE_loss/000/checkpoints/last_checkpoint.pth"
# COMPOSITION_PATH="./datasets/Composition-1k-testset"
# P3M500_PATH="./datasets/2.P3M-10k/P3M-10k/validation/P3M-500-NP"
# AIM500_PATH="./datasets/AIM-500"
# AM200_PATH="./datasets/AM-200"

# LOG_DIR="./evaluation/eval_logs"

# python evaluation/trimap/eval.py \
#     --weight_path "$WEIGHT_PATH" \
#     --device "cpu" \
#     --batch_size 20 \
#     --log_dir "$LOG_DIR" \
#     --composition_path "$COMPOSITION_PATH" \
#     --p3m500_path "$P3M500_PATH" \
#     --aim500_path "$AIM500_PATH" \
#     --am200_path "$AM200_PATH"
#kjk20250508


# MODEL_PATH=./weights/simpleclick_models/cocolvis_vit_base.pth

# python scripts/evaluate_model.py NoBRS \
# --gpu=0 \
# --checkpoint=${MODEL_PATH} \
# --eval-mode=cvpr \
# --datasets=GrabCut
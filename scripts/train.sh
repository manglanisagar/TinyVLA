#!/bin/bash

ACTION_HEAD=droid_diffusion
OUTPUT="llava_pythia"

if [ -d "$OUTPUT" ]; then
   echo 'Output exists'
else
   echo '!! Output does not exist !!'
   mkdir -p $OUTPUT
fi

cp ./scripts/train.sh $OUTPUT

export CUDA_VISIBLE_DEVICES=0  # set only if single GPU

accelerate launch --deepspeed_config_file scripts/zero2.json ./train_tinyvla.py \
  --lora_enable True \
  --lora_module 'vit llm' \
  --load_pretrain False \
  --pretrain_image_size 320 \
  --lora_r 64 \
  --lora_alpha 256 \
  --non_lora_lr 2e-5 \
  --task_name "spot_dummy_test" \
  --model_name_or_path "/home/sagar/projects/TinyVLA/Llava-Pythia-400M" \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower True \
  --freeze_backbone True \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 1000 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 50 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.005 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --action_head_type $ACTION_HEAD \
  --concat "token_cat" \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log

# Copy preprocessor config to all checkpoints
for dir in "$OUTPUT"/*/ ; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp llava-pythia/preprocessor_config.json "$dir"
    fi
done


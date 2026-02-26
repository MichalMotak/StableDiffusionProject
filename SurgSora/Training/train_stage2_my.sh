

EXP_NAME="train_stage2_cataract5"

accelerate launch --main_process_port 29800 --num_processes=2   train_stage2_my.py \
 --pretrained_model_name_or_path="ckpts/stable-video-diffusion-img2vid-xt-1-1"\
 --controlnet_model_name_or_path="logs/train_stage1_cataract4/checkpoint-26400/controlnet" \
 --output_dir="logs/${EXP_NAME}/" \
 --height=256 \
 --width=256 \
 --train_height=256 \
 --train_width=256 \
 --seed=42 \
 --learning_rate=2e-5 \
 --per_gpu_batch_size=1 \
 --num_train_epochs=100 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=1 \
 --checkpoints_total_limit=100 \
 --gradient_checkpointing \
 --num_validation_images=2 \
 --use_8bit_adam \
 --sample_stride=4 \
 --num_frames=21 \


 #  --max_train_steps=50000 \
#  --checkpointing_steps=1000 \
#  --validation_steps=1000 \
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

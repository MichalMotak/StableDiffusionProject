# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# export TMPDIR=/home/MichalMo/tmp
# mkdir -p /home/MichalMo/tmp


EXP_NAME="train_stage1_cataract_789_ttttt"

accelerate launch --main_process_port 29700 --num_processes=1 train_stage1_my.py \
 --pretrained_model_name_or_path="ckpts/stable-video-diffusion-img2vid-xt-1-1"\
 --output_dir="logs/${EXP_NAME}/" \
 --width=256 \
 --height=256 \
 --seed=42 \
 --learning_rate=2e-5 \
 --per_gpu_batch_size=1 \
 --num_train_epochs=50 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=1 \
 --checkpoints_total_limit=50 \
 --num_frames=21 \
 --gradient_checkpointing \
 --num_validation_images=2 \
 --sample_stride=1 \

#  --max_train_steps=None \
#  --checkpointing_steps=50 \
# --validation_steps=50 \

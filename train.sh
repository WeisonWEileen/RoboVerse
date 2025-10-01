# train with walking

python ./humanoid_visualrl/scripts/train.py \
    --num_envs 64 \
    --task "active_vision" \
    --run_name "refactor_cnn" \
    --enable_opencv_display 
    # --wandb \
    # --resume \
    # --load_run "2025_0918_064216" \
    # --checkpoint 4400 \
    # --use_fixed_gazing \
    # --use_vision \
    # --resume 
    # --headless \
    # --load_run "2025_0905_013206" \
    # --checkpoint 556 \



# for pixel gaze reward and resnet18 encoder
# python ./humanoid_visualrl/scripts/train.py \
#     --headless \
#     --num_envs 1 \
#     --use_fixed_gazing \
#     --robot g1_static \
#     --wandb



# for pixel gaze reward and cnn encoder
# python ./humanoid_visualrl/scripts/train.py \
#     --headless \
#     --num_envs 512 \
#     --use_fixed_gazing \
#     --use_vision \
#     --robot g1_static \
#     --enable_opencv_display \
#     --wandb


# train with walking

python ./humanoid_visualrl/scripts/train.py \
    --num_envs 256 \
    --headless \
    --wandb \
    --robot "g1_static" \
    --use_fixed_gazing \
    --use_vision \
    --resume \
    --load_run "2025_0905_013206" \
    --checkpoint 556 \

    # --enable_opencv_display \


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

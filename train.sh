# train with walking

python ./humanoid_visualrl/scripts/train.py \
    --num_envs 1 \
    # --headless \
    # --enable_opencv_display \
    # --wandb


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

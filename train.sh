# train with walking

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
# /home/balen/conda/envs/metasim/bin/python ./humanoid_visualrl/scripts/train.py \
USER_NAME=$(whoami)

if [ "$USER_NAME" = "balen" ]; then
    PYTHON_PATH="/home/balen/conda/envs/metasim/bin/python"
    echo "Using balen's python path"
elif [ "$USER_NAME" = "ghr" ]; then
    PYTHON_PATH="/datasets/v2p/current/pw-workspace/conda/isaaclab211/bin/python"
    echo "Using ghr's python path"
else
    PYTHON_PATH="python3"
    echo "Using default python path"
fi

$PYTHON_PATH ./humanoid_visualrl/scripts/train.py \
    --num_envs  64 \
    --task "active_vision" \
    --run_name "curriculum_vision_cube_entropy_smaller" \
    --device "cuda:0" \
    --wandb \
    --enable_opencv_display \
    # --headless 
    # --headless
    # --debug

    # --wandb
    # --headless \
    # --wandb \
    # --resume \
    # --load_run "2025_1005_004654" \
    # --checkpoint 3200 \
    # --wandb
    # --wandb \
    # --debug
    # --debug
    # --wandb
    # --debug 
    # --wandb 
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


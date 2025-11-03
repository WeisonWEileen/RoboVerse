# train with walking
export WANDB_API_KEY=70b35cc989ebf8652e52516c433f9faa444d21d2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
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
    --num_envs 128 \
    --task "active_vision" \
    --run_name "reaching_ckpt_2025_1102_120613_scale_look1.4_reach0.4" \
    --device "cuda:0" \
    --enable_opencv_display \
    --wandb \
    --load_run "2025_1102_120613" \
    --resume \
    --checkpoint 400 \
    # --debug
    # --wandb \
    # --headless 
    # --wandb \
    # --headless 
    # --debug
    # --enable_opencv_display \
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


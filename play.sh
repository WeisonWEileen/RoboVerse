# python ./humanoid_visualrl/scripts/play.py \
#     --num_envs 1 \
#     --robot "g1_static" \
#     --use_fixed_gazing \
#     --use_vision \
#     --resume \
#     --load_run "2025_0905_043450" \
#     --checkpoint 700 \
#     --enable_opencv_display 
#     # --wandb \


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

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

# $PYTHON_PATH ./humanoid_visualrl/scripts/evaluate_see_reaching_sr.py \
$PYTHON_PATH ./humanoid_visualrl/scripts/evaluate_see_sr.py \
    --task "active_vision" \
    --resume \
    --load_run "2025_1108_043702" \
    --checkpoint 600 \
    --headless \
    --device "cuda:3" \
    # --enable_opencv_display 
    # --headless \
    # --wandb \

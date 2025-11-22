# train with reaching skill
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
    --num_envs 3096 \
    --task "walking" \
    --run_name "walking" \
    --robot "g1_pp_comp" \
    --device "cuda:0" \
    --wandb 

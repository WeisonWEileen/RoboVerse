# train with walking
export WANDB_API_KEY=70b35cc989ebf8652e52516c433f9faa444d21d2
export CUDA_VISIBLE_DEVICES=0
# /home/balen/conda/envs/metasim/bin/python ./humanoid_visualrl/scripts/train.py \
USER_NAME=$(whoami)

if [ "$USER_NAME" = "balen" ]; then
    PYTHON_PATH="/home/balen/conda/envs/metasim/bin/python"
    echo "Using balen's python path"
elif [ "$USER_NAME" = "ghr" ]; then
    PYTHON_PATH="/datasets/v2p/current/pw-workspace/conda/isaaclapb211/bin/python"
    echo "Using ghr's python path"
elif [ "$USER_NAME" = "haoran" ]; then
    PYTHON_PATH="/home/haoran/miniconda3/envs/active_vision/bin/python"
    echo "Using haoran's python path"
else
    PYTHON_PATH="python3"
    echo "Using default python path"
fi

$PYTHON_PATH ./humanoid_visualrl/scripts/train.py \
    --num_envs 128 \
    --task "active_vision_cube" \
    --run_name "insertion_phase2" \
    --device "cuda:0" \
    --actor_critic_class "use_rnn_cnn_ram" \
    --enable_opencv_display \
    --phase2 \
    --wandb \
    # --resume \
    # --checkpoint 100 \
    # --load_run 2026_0130_004031 
    # --phase 2 

    # --resume \
    # --checkpoint 2700 \
    # --load_run "2026_0106_023609" \
    # --randomize_material \
    # --actor_critic_class "use_rnn_cnn_ram" \

    # --vision4times_slowdown \
    # --debug
    # --wandb \
    # --enable_grasp \
    # --debug



    # --schedule "momentum" 
    # --seed 42 \
    
    # --headless
    # --headless \
    # --randomize_material \
    # --debug
    
    # --wandb \
    # --headless \
    # --debug
    # --run_name "pretrain_material_kl_clipping_1.1" \
    # --load_run "2025_1107_132622" \
    # --resume \
    # --checkpoint 800 \
    # --run_name "layernorm_critic_with_pretrained_400_searching" \
    # --enable_opencv_display \
    # --wandb \
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


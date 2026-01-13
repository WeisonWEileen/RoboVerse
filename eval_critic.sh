python ./humanoid_visualrl/scripts/evaluate_critic.py \
    --task "active_vision" \
    --device "cuda:0" \
    --actor_critic_class "use_rnn_cnn_ram" \
    --resume \
    --checkpoint 1400 \
    --load_run 2026_0112_030824
BASE_PATH="/home/zxliang/new-code/LISA/lisa/outputs/2023-03-09/20-56-04/checkpoints/LorlEnv-v0-40108-traj_option-2023-03-09-20:56:04"
ITER=500
CKPT="${BASE_PATH}/model_${ITER}.ckpt"

#method=option_dt
python hrl/main.py env=lorel_sawyer_obs method=traj_option dt.n_layer=1 dt.n_head=4 option_selector.option_transformer.n_layer=1 option_selector.option_transformer.n_head=4 option_selector.commitment_weight=0.1 option_selector.option_transformer.hidden_size=128 batch_size=256 seed=1 warmup_steps=5000 eval=True render=True checkpoint_path=${CKPT}

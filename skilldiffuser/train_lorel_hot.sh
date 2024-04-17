python hrl/main_hot.py env=lorel_sawyer_obs method=traj_option dt.n_layer=1 dt.n_head=4 option_selector.option_transformer.n_layer=1 option_selector.option_transformer.n_head=4 option_selector.commitment_weight=0.1 option_selector.option_transformer.hidden_size=128 batch_size=256 seed=1 warmup_steps=5000 resume=True checkpoint_path="/data/zxliang/new-code/LISA-hot/lisa/outputs/2024-01-25/15-06-21/checkpoints/LorlEnv-v0-40108-traj_option-2024-01-25-15:06:25/model_100.ckpt" diffuser.loadpath="/data/zxliang/new-code/LISA-hot/lisa/outputs/2024-01-25/15-06-21/buckets/diffuser_log/checkpoint/state_160000.pt"

#checkpoint_path="/home/zxliang/new-code/LISA-diffuser/lisa/outputs/2023-09-28/08-52-47/checkpoints/LorlEnv-v0-40108-traj_option-2023-09-28-08:52:47/model_300.ckpt" diffuser.loadpath="/home/zxliang/new-code/LISA-diffuser/lisa/outputs/2023-09-28/08-52-47/buckets/diffuser_log/checkpoint/state_480000.pt"

#checkpoint_path="/home/zxliang/new-code/LISA-diffuser/lisa/outputs/2023-09-30/07-15-13/checkpoints/LorlEnv-v0-40108-traj_option-2023-09-30-07:15:13/model_250.ckpt" diffuser.loadpath="/home/zxliang/new-code/LISA-diffuser/lisa/outputs/2023-09-30/07-15-13/buckets/diffuser_log/checkpoint/state_400000.pt"

# checkpoint_path="/home/zxliang/new-code/LISA/lisa/outputs/2023-10-19/10-32-01/checkpoints/LorlEnv-v0-40108-traj_option-2023-10-19-10:32:02/model_350.ckpt" diffuser.loadpath="/home/zxliang/new-code/LISA/lisa/outputs/2023-10-19/10-32-01/buckets/diffuser_log/checkpoint/state_460000.pt"

#checkpoint_path="/data/zxliang/new-code/LISA-hot/lisa/outputs/2024-01-25/15-06-21/checkpoints/LorlEnv-v0-40108-traj_option-2024-01-25-15:06:25/model_100.ckpt" diffuser.loadpath="/data/zxliang/new-code/LISA-hot/lisa/outputs/2024-01-25/15-06-21/buckets/diffuser_log/checkpoint/state_160000.pt"

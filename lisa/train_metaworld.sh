python hrl/main.py env=metaworld method=traj_option dt.n_layer=1 dt.n_head=4 option_selector.option_transformer.n_layer=1 option_selector.option_transformer.n_head=4 option_selector.commitment_weight=0.1 option_selector.option_transformer.hidden_size=128 batch_size=64 seed=1 warmup_steps=5000
#resume=True checkpoint_path="/home/zxliang/new-code/LISA4MT/lisa/outputs/2024-01-25/12-08-52/checkpoints/MetaWorld-MT10-v2-1000-traj_option-2024-01-25-12:08:52/model_300.ckpt"


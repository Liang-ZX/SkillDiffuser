import pdb
import sys

try:
    sys.path.append("..")
    import babyai
except:
    print('BabyAI env is not installed')

try:
    import lorl_env
except:
    print('Lorl env not installed')

import gym
import numpy as np
import torch
import wandb
import datetime
import os
import ast
import hydra
import random
from omegaconf import DictConfig, OmegaConf
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from hrl_model import HRLModel
from trainer import Trainer


def evaluate(cfg):
    # load saved arguments
    checkpoint = torch.load(cfg.checkpoint_path)
    args = checkpoint['config']
    max_length = checkpoint['train_dataset_max_length']
    args.eval = cfg.eval
    args.render = cfg.render
    args.checkpoint_path = cfg.checkpoint_path
    device = cfg.trainer.device

    # Set num train_trajs to something small
    args.train_dataset.num_trajectories = 1000
    print(OmegaConf.to_yaml(args))

    args.method = args.model.name

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    num_eval_episodes = args.trainer.num_eval_episodes
    print('=' * 50)
    print(f'Starting evaluation: {args.env.name}')
    print(f'{args.trainer.num_eval_episodes} trajectories')
    print('=' * 50)

    state_dim = args.env.state_dim
    action_dim = args.env.action_dim
    if isinstance(state_dim, str):
        state_dim = ast.literal_eval(state_dim)

    if isinstance(state_dim, tuple):
        assert not args.trainer.state_il, "Cannot do state imitation learning with an image input"

    if not args.env.eval_offline:
        env = gym.make(args.env.name)
        env_name = args.env.name
        env.seed(args.seed)

    if 'BabyAI' in args.env.name:
        state_dim += 4*args.env.use_direction

    train_dataset_args = dict(args.train_dataset)
    batch_size = args.batch_size

    if 'BabyAI' in args.env.name:
        train_dataset = ExpertDataset(**train_dataset_args, use_direction=args.env.use_direction)
    elif 'Lorl' in args.env.name or 'MetaWorld' in args.env.name:
        # train_dataset_args also contains a split here for the validation data size
        train_dataset = ExpertDataset(**train_dataset_args, use_state=args.env.use_state)
    else:
        raise NotImplementedError
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=32,
                              shuffle=True, drop_last=True)

    if args.method == 'traj_option':
        args.option_selector.option_transformer.max_length = int(max_length)
        args.option_selector.option_transformer.max_ep_len = args.env.eval_episode_factor * \
            int(max_length)

    option_selector_args = dict(args.option_selector)
    option_selector_args['state_dim'] = state_dim
    option_selector_args['option_dim'] = args.option_dim
    option_selector_args['codebook_dim'] = args.codebook_dim
    option_selector_args['env_name'] = args.env.name
    state_reconstructor_args = dict(args.state_reconstructor)
    lang_reconstructor_args = dict(args.lang_reconstructor)
    decision_transformer_args = {'state_dim': state_dim,
                                 'action_dim': action_dim,
                                 'option_dim': args.option_dim,
                                 'discrete': args.env.discrete,
                                 'hidden_size': args.dt.hidden_size,
                                 'use_language': args.method == 'vanilla',
                                 'use_options': args.method != 'vanilla',
                                 'option_il': args.dt.option_il,
                                 'max_length': max_length if args.method != 'traj_option' else args.model.K,
                                 'max_ep_len': args.env.eval_episode_factor*max_length,
                                 'action_tanh': False,
                                 'n_layer': args.dt.n_layer,
                                 'n_head': args.dt.n_head,
                                 'n_inner': 4*args.dt.hidden_size,
                                 'activation_function': args.dt.activation_function,
                                 'n_positions': args.dt.n_positions,
                                 'n_ctx': args.dt.n_positions,
                                 'resid_pdrop': args.dt.dropout,
                                 'attn_pdrop': args.dt.dropout,
                                 }
    hrl_model_args = dict(args.model)

    iq_args = cfg.iq

    model = HRLModel(option_selector_args, state_reconstructor_args,
                     lang_reconstructor_args, decision_transformer_args, iq_args, device, **hrl_model_args)

    print(model)
    model = model.to(device=device)

    trainer_args = dict(args.trainer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=None,
        train_loader=train_loader,
        env=env,
        env_name=env_name,
        val_loader=None,
        scheduler=None,
        skip_words=args.env.skip_words,
        **trainer_args
    )

    # Restore trainer from checkpoint
    trainer.load(args.checkpoint_path)
    trainer.evaluate(iter_num=0, render=args.render, max_ep_len=500, render_path=args.render_path)


def train(args):
    device = args.trainer.device

    args.method = args.model.name
    exp_name = f'{args.project_name}-{args.train_dataset.num_trajectories}-{args.method}'
    args.savepath = f'{args.hydra_base_dir}/{args.savedir}/{exp_name}-{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'

    if args.wandb:
        wandb.init(
            name=exp_name,
            group=args.method,
            project=f'debug_LISA_metaworld_MT_pluo_{args.env.name}',  # TODO frozen  diff_hot_A100_LN
            config=dict(args),
            # entity='language-rl',
            # mode="offline",
        )

    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath, exist_ok=True)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    #K = args['K']
    batch_size = args.batch_size

    train_dataset_args = dict(args.train_dataset)
    if 'BabyAI' in args.env.name:
        train_dataset = ExpertDataset(**train_dataset_args, use_direction=args.env.use_direction)
    elif 'Lorl' in args.env.name or 'MetaWorld' in args.env.name:
        train_dataset = ExpertDataset(**train_dataset_args, use_state=args.env.use_state)
    elif 'Hopper' in args.env.name:
        train_dataset = ExpertDataset(**train_dataset_args)
    else:
        raise NotImplementedError
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=32,
                              shuffle=True, pin_memory=True, drop_last=True)

    print('=' * 50)
    print(f'Starting new experiment: {args.env.name} {args.train_dataset.num_trajectories}')
    print(f'{len(train_dataset)} trajectories, {train_dataset.total_timesteps} timesteps found')
    print('=' * 50)

    state_dim = args.env.state_dim
    action_dim = args.env.action_dim
    if isinstance(state_dim, str):
        state_dim = ast.literal_eval(state_dim)
        args.env.state_dim = state_dim

    if isinstance(state_dim, tuple):
        assert not args.trainer.state_il, "Cannot do state imitation learning with an image input"

    if not args.env.eval_offline:
        if args.env.eval_env:
            env_name = args.env.eval_env
        else:
            env_name = args.env.name
        print(f'-->Testing on {env_name}')

        if 'MetaWorld' in args.env.name:
            env = "MetaWorld"
            env_name = args.env.name
        else:
            env = gym.make(env_name)
            env_name = env_name
        val_loader = None
    else:
        val_dataset_args = dict(args.val_dataset)
        if 'BabyAI' in args.env.name:
            val_dataset = ExpertDataset(**val_dataset_args, use_direction=args.env.use_direction)
        elif 'Lorl' in args.env.name or 'MetaWorld' in args.env.name:
            val_dataset = ExpertDataset(**val_dataset_args, use_state=args.env.use_state)
        else:
            raise NotImplementedError
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=32,
                                shuffle=True, pin_memory=True, drop_last=True)

    if 'BabyAI' in args.env.name:
        state_dim += 4*args.env.use_direction

    print(f'--> Train episode max length: {train_dataset.max_length}')
    if args.method == 'traj_option':
        args.option_selector.option_transformer.max_length = int(train_dataset.max_length)
        args.option_selector.option_transformer.max_ep_len = args.env.eval_episode_factor * \
            int(train_dataset.max_length)

    if args.model.horizon == 'max':
        args.model.horizon = int(train_dataset.max_length)
    if args.model.K == 'max':
        args.model.K = int(train_dataset.max_length)

    option_selector_args = dict(args.option_selector)
    option_selector_args['state_dim'] = state_dim
    option_selector_args['option_dim'] = args.option_dim
    option_selector_args['codebook_dim'] = args.codebook_dim
    option_selector_args['env_name'] = args.env.name
    state_reconstructor_args = dict(args.state_reconstructor)
    lang_reconstructor_args = dict(args.lang_reconstructor)
    decision_transformer_args = {'state_dim': state_dim,
                                 'action_dim': action_dim,
                                 'option_dim': args.option_dim,
                                 'discrete': args.env.discrete,
                                 'hidden_size': args.dt.hidden_size,
                                 'use_language': args.method == 'vanilla',
                                 'use_options': args.method != 'vanilla',
                                 'option_il': args.dt.option_il,
                                 'predict_q': args.use_iq,
                                 'max_length': train_dataset.max_length if 'option' not in args.method else args.model.K,   # used to be K
                                 'max_ep_len': args.env.eval_episode_factor*train_dataset.max_length,
                                 'n_layer': args.dt.n_layer,
                                 'n_head': args.dt.n_head,
                                 'activation_function': args.dt.activation_function,
                                 'n_positions': args.dt.n_positions,
                                 'n_ctx': args.dt.n_positions,
                                 'resid_pdrop': args.dt.dropout,
                                 'attn_pdrop': args.dt.dropout,
                                 'no_states': args.dt.no_states,
                                 'no_actions': args.dt.no_actions,
                                 }
    hrl_model_args = dict(args.model)
    iq_args = args.iq

    model = HRLModel(args, option_selector_args, state_reconstructor_args,
                     lang_reconstructor_args, decision_transformer_args, iq_args, device, **hrl_model_args)

    start_iter = 1
    if args.resume:
        args.warmup_steps = 0
        #checkpoint = trainer.load(args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        # start_iter = checkpoint['iter_num'] + 1
        assert train_dataset.max_length == checkpoint[
            'train_dataset_max_length'], f"Expected max length of dataset to be {train_dataset.max_length} but got {checkpoint['train_dataset_max_length']}"
        
    if args.load_options:
        checkpoint = torch.load(args.checkpoint_path)
        checkpoint = checkpoint['model']
        state_dict = {k:v for k,v in checkpoint.items() if k.startswith('option_selector.Z')}
        loaded = model.load_state_dict(state_dict, strict=False)
        assert loaded.unexpected_keys == []   ## simple check
        if args.freeze_loaded_options:
            for name, param in model.named_parameters():
                if name.startswith('option_selector.Z'):
                    param.requires_grad = False
            assert not model.option_selector.Z.project_out.bias.requires_grad   ## simple check

    if args.parallel:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device=device)

    # Setting up the optimizer
    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    # setting different learning rates for the LM part, OS part and other parts
    lm_params = {'params': [v for k, v in params if k.startswith('lm.') or k.startswith('module.lm.')], 'lr': args.lm_learning_rate}
    os_params = {'params': [v for k, v in params if k.startswith('option_selector.') or k.startswith('module.option_selector.')], 'lr': args.os_learning_rate}
    other_params = {'params': [v for k, v in params if not k.startswith(
        'lm.') and not k.startswith('option_selector.')
        and not k.startswith('module.lm.') and not k.startswith('module.option_selector.')]}
    # for the option selector need separate lr?
    optimizer = torch.optim.AdamW(
        [other_params, lm_params, os_params],
        lr=args.learning_rate, weight_decay=args.weight_decay,)

    def adjust_lr(steps):
        if steps < args.warmup_steps:
            return min((steps + 1) / args.warmup_steps, 1)
        num_decays = (steps + 1) // args.decay_steps
        return args.lr_decay ** (num_decays)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, adjust_lr)

    trainer_args = dict(args.trainer)

    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_loader=train_loader,
        env=env,
        env_name=env_name,
        val_loader=val_loader,
        scheduler=scheduler,
        eval_episode_factor=2,
        skip_words=args.env.skip_words,
        **trainer_args
    )

    # Training loop
    for iter_num in range(start_iter, start_iter + args.max_iters):
        outputs = trainer.train_iteration(
            iter_num=iter_num, print_logs=True, eval_render=args.render)

        if args.wandb and iter_num % args.log_interval == 0:
            wandb.log(outputs, step=iter_num)

        if iter_num % args.save_interval == 0:
            trainer.save(iter_num, f'{args.savepath}/model_{iter_num}.ckpt', args)


def get_args(cfg: DictConfig):
    cfg.trainer.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    return cfg


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    args = get_args(cfg)

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.trainer.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif device.type == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("--> Running in ", os.getcwd())

    if args.eval:
        evaluate(cfg)
        return

    # train
    print(OmegaConf.to_yaml(cfg))
    train(args)


if __name__ == "__main__":
    main()

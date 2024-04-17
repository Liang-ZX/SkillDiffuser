#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os
import imageio
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] =" 4"


# In[3]:


import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf


# In[4]:


## WITH 20 options lorl states and horizon 10

# BASE_PATH="/atlas/u/divgarg/projects/Language-RL/hrl/outputs/2021-12-18/21-35-16/checkpoints/LorlEnv-v0-40108-traj_option-2021-12-18-21:35:16/"
# ITER = 360


# In[5]:


## WITH 20 OPTIONS LORL IMAGES, HORIZON 10

# BASE_PATH = "/atlas/u/divgarg/projects/Language-RL/hrl/outputs/2021-12-22/11-59-46/checkpoints/LorlEnv-v0-40108-traj_option-2021-12-22-11:59:46"
# ITER = 500


# In[6]:


## WITH 20 OPTIONS LORL IMAGES, HORIZON 10

#BASE_PATH = "/atlas/u/divgarg/projects/Language-RL/hrl/outputs/2022-01-07/14-24-30/checkpoints/LorlEnv-v0-40108-traj_option-2022-01-07-14:24:31"
#ITER = 70 # 20, 70
# eval_episode_factor = 10


# In[7]:


#BASE_PATH = "/atlas/u/divgarg/projects/Language-RL/hrl/outputs/2022-01-13/11-21-27/checkpoints/LorlEnv-v0-40108-traj_option-2022-01-13-11:21:28"
BASE_PATH = "/atlas/u/divgarg/projects/Language-RL/hrl/outputs/2022-01-13/11-25-35/checkpoints/LorlEnv-v0-40108-vanilla-2022-01-13-11:25:35"
ITER = 500


# In[8]:


CKPT=f"{BASE_PATH}/model_{ITER}.ckpt"


# In[9]:


with initialize("conf"):
    cfg = compose(config_name="config.yaml", overrides=["eval=True", "method=option_dt", f"checkpoint_path={CKPT}"])
    print(cfg)


# In[10]:


from main import *
import ast

def evaluate(cfg):
    # load saved arguments
    checkpoint = torch.load(cfg.checkpoint_path)
    args = checkpoint['config']
    max_length = checkpoint['train_dataset_max_length']
    args.eval = cfg.eval
    args.render = cfg.render
    args.checkpoint_path = cfg.checkpoint_path
    device = cfg.trainer.device

    # args.env.eval_episode_factor = 1
    # Set num train_trajs to something small
    args.train_dataset.num_trajectories = 1000
    print(OmegaConf.to_yaml(args))

    args.method = args.model.name

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    num_eval_episodes = args.trainer.num_eval_episodes

    if not args.env.eval_offline:
        env = gym.make(args.env.name)
        #env.seed(args.seed)
        state_dim = args.env.state_dim
        if isinstance(state_dim, str):
            state_dim = ast.literal_eval(state_dim)
        action_dim = args.env.action_dim

    if 'BabyAI' in args.env.name:
        state_dim += 4*args.env.use_direction

    train_dataset_args = dict(args.train_dataset)
    batch_size = args.batch_size

    if 'BabyAI' in args.env.name:
        train_dataset = ExpertDataset(**train_dataset_args, use_direction=args.env.use_direction)
    elif 'Lorl' in args.env.name:
        # train_dataset_args also contains a split here for the validation data size
        train_dataset = ExpertDataset(**train_dataset_args, use_state=args.env.use_state)
    else:
        raise NotImplementedError
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=32,
                              shuffle=True, drop_last=True)
    del train_dataset
    
    eval_episode_factor =  args.env.eval_episode_factor 
    
    if args.method == 'traj_option':
        args.option_selector.option_transformer.max_length = int(max_length)
        args.option_selector.option_transformer.max_ep_len = eval_episode_factor * int(max_length)
        # args.option_selector.option_transformer.output_attention = True

    option_selector_args = dict(args.option_selector)
    option_selector_args['state_dim'] = state_dim
    option_selector_args['option_dim'] = args.option_dim
    option_selector_args['codebook_dim'] = args.codebook_dim
    # option_selector_args['option_transformer']['output_attention'] = True
    
    state_reconstructor_args = dict(args.state_reconstructor)
    lang_reconstructor_args = dict(args.lang_reconstructor)
    decision_transformer_args = {'state_dim': state_dim,
                                 'action_dim': action_dim,
                                 'option_dim': args.option_dim,
                                 'discrete': args.env.discrete,
                                 'hidden_size': args.dt.hidden_size,
                                 'use_language': args.method == 'vanilla',
                                 'use_options': args.method != 'vanilla',
                                 'max_length': max_length if args.method != 'traj_option' else args.model.K,
                                 # setting this to be sufficiently large so that there is enough of a buffer during eval
                                 'max_ep_len': eval_episode_factor*max_length,
                                 'action_tanh': False,
                                 'n_layer': args.dt.n_layer,
                                 'n_head': args.dt.n_head,
                                 'n_inner': 4*args.dt.hidden_size,
                                 'activation_function': args.dt.activation_function,
                                 'n_positions': args.dt.n_positions,
                                 'resid_pdrop': args.dt.dropout,
                                 'attn_pdrop': args.dt.dropout,
                                 }
    hrl_model_args = dict(args.model)
    iq_args = cfg.iq

    model = HRLModel(option_selector_args, state_reconstructor_args,
                     lang_reconstructor_args, decision_transformer_args, iq_args, device, **hrl_model_args)

    model = model.to(device=device)

    trainer_args = dict(args.trainer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=None,
        train_loader=train_loader,
        env=env,
        val_loader=None,
        scheduler=None,
        **trainer_args
    )

    # Restore trainer from checkpoint
    ## TEMP: DISABLE LOADING CHECKPOINT
    trainer.load(args.checkpoint_path)
    return model, tokenizer, train_loader, env, args


# In[11]:


model, tokenizer, train_loader, env, args = evaluate(cfg)
#max_length = args.option_selector.option_transformer.max_length


# In[12]:


# from typing import Dict, Iterable, Callable
# import torch.nn as nn
# from torch import Tensor

# class Attention(nn.Module):
#     def __init__(self, model: nn.Module):
#         super().__init__()
#         self.model = model
#         self._attention = None

#         model.option_selector.option_dt.register_forward_hook(self.save_attention_hook())

#     def save_attention_hook(self) -> Callable:
#         def fn(model, input, output):
#             self._attention= output[-1]
#         return fn

#     def forward(self, x: Tensor) -> Dict[str, Tensor]:
#         _ = self.model(x)
#         return self._attention


# In[13]:


#att_model = Attention(model)


# In[14]:


from PIL import Image
from env import LorlWrapper, BabyAIWrapper
from eval import get_action

def traj(env, train_loader, model, tokenizer, args, render=False, render_freq=1, instr=None, orig_instr=None, **kwargs):
    
    if env:
        if 'BabyAI' in args.env.name:
            env = BabyAIWrapper(env, train_loader.dataset)
            # For BabyAI env
            max_ep_len = 300  # 2 * train_loader.dataset.max_length

        
        elif 'Lorl' in args.env.name:
            env = LorlWrapper(env, train_loader.dataset, instr=instr, orig_instr=orig_instr)
            max_ep_len = 60#40 # args.env.eval_episode_factor * train_loader.dataset.max_length - 1 
            
    model.eval()

    if hasattr(model, 'module'):
        model = model.module
    else:
        model = model

    device = args.trainer.device
    method = model.method
    K = args.model.K  # 20
    horizon = model.horizon  # 20
    option_dim = model.option_dim
    model = model.to(device=device)
        
    if env:
        if method != 'vanilla':
            option_dim = model.option_selector.option_dim

        returns, lengths, successes = [], [], []
        observation = env.reset()
        state, lang = observation['state'], observation['lang']

        state_dim = env.state_dim
        act_dim = env.act_dim
        cur_state = torch.from_numpy(state)

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        lm_input = tokenizer(text=[lang], add_special_tokens=True,
                                  return_tensors='pt', padding=True).to(device=device)
        with torch.no_grad():
            lm_embeddings = model.lm(
                lm_input['input_ids'], lm_input['attention_mask']).last_hidden_state
            cls_embeddings = lm_embeddings[:, 0, :]
            word_embeddings = lm_embeddings[:, 1:, :]      # skip the CLS and SEP tokens. here there's no padding so this is actually the CLS and SEP
            # word_embeddings = lm_embeddings
        
        if isinstance(state_dim, tuple):
            states = torch.from_numpy(state).reshape(1, *state_dim).to(device=device, dtype=torch.float32)
        else:
            states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        if method != 'vanilla':
            options = torch.zeros((0, option_dim), device=device, dtype=torch.float32)

        episode_return, episode_length, success = 0, 0, 0
        options_list = []
        images = []

        option = None

        for t in range(max_ep_len):
            # add dummy action
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            if method != 'vanilla':
                options = torch.cat([options, torch.zeros((1, option_dim), device=device)], dim=0)
            else:
                options = None

            action, option, states, actions, timesteps, options = get_action(
                model, states, actions, options, timesteps, cls_embeddings, word_embeddings, options_list, cur_state,
                option, t, horizon, K, method, state_dim, act_dim, option_dim, device, **kwargs)

            if model.decision_transformer.discrete:
                actions[-1] = torch.nn.functional.one_hot(action, num_classes=act_dim)
            else:
                action = torch.clamp(action, torch.from_numpy(env.action_space.low).to(
                    device), torch.from_numpy(env.action_space.high).to(device))
                actions[-1] = action

            action = action.detach().cpu().numpy()
            assert action in env.action_space, "Transformer predicted action outside env action space"

            obs, reward, done, info = env.step(action)
            if render:
                images.append(env.get_image())

            state, lang = obs['state'], obs['lang']
            if isinstance(state_dim, tuple):
                cur_state = torch.from_numpy(state).to(device=device).reshape(1, *state_dim)
            else:
                cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0).float()
            timesteps = torch.cat(
                [timesteps,
                    torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                success = info['success']
                break
            
        if render:
            imageio.mimsave(f'gifs/{instr}.gif', images)

    return  word_embeddings, states.to(dtype=torch.float32), timesteps.to(dtype=torch.long), images, lm_input, options_list, episode_length, success


# In[15]:


import random
from utils import LORL_EVAL_INSTRS, LORL_COMPOSITION_INSTRS
import copy

# kwargs = {}
# if 'Lorl' in args.env.name:
#     orig_instr, rephrasals = random.choice(list(LORL_EVAL_INSTRS.items()))
#     rephrasal_type, instr_list = random.choice(list(rephrasals.items()))
#     instr = random.choice(instr_list)
#     kwargs['orig_instr'] = orig_instr
#     kwargs['instr'] = instr
                        
# words, states, timesteps, images, lm_inputs, options_list, episode_length, success = traj(env, train_loader, model, tokenizer, args, **kwargs)
# print(f'Generated episode of length: {episode_length}')
# print(f'Generated episode success: {success}')


# In[16]:


# att_model.model.option_selector.get_option(words, states, timesteps)


# In[17]:


#!pip install bertviz wordcloud


# In[18]:


# from bertviz import head_view, model_view
# from transformers import BertTokenizer, BertModel

# def get_tokens(inputs):
#     # model_version = 'bert-base-uncased'
#     # do_lower_case = True
#     # model = BertModel.from_pretrained(model_version, output_attentions=True)
#     # tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
#     # sentence_a = "The cat sat on the mat"
#     # sentence_b = "The cat lay on the rug"
#     # inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)

#     input_ids = inputs['input_ids']
#     # token_type_ids = inputs['token_type_ids']
#     # attention = model(input_ids, token_type_ids=token_type_ids)[-1]
#     # sentence_b_start = token_type_ids[0].tolist().index(1)
#     input_id_list = input_ids[0].tolist() # Batch index 0
#     tokens = tokenizer.convert_ids_to_tokens(input_id_list)[1:] 
#     return tokens


# In[19]:


# tokens = get_tokens(lm_inputs)
# N = len(tokens)


# In[20]:


# attention = att_model._attention
# L = min(episode_length, max_length)
# P = max_length -  L
# idx = [True] * N + [False] * P + [True] * L
# print(N, P, L)
# out = []

# for layer in range(len(attention)):
#     x = attention[layer][:, :, idx, :]
#     x = x[:, :, :, idx]
#     out.append(x)
# print(len(out))
# print(out[-1].shape)


# In[21]:


# Load model and retrieve attention weights
# options = np.repeat(options_list, args.model.horizon)[:L]

# out[-1].shape


# In[22]:


# options_list


# In[23]:


#head_view(out, tokens + [f's_{i}, o_{o}' for i, o in enumerate(options)], layer=5)
# head_view(out, tokens + [f's_{i}, o_{o}' for i, o in enumerate(options)], layer=args.option_selector.option_transformer.n_layer-1)


# In[24]:


# import matplotlib.pyplot as plt

# for i, im in enumerate(images):
#     print(f'state: {i}')
#     plt.figure(figsize = (8,8))
#     plt.imshow(im)
#     plt.show()


# In[25]:


# Visualize word bags


# In[26]:


# from tqdm import tqdm

# num_options = 0 if args.method == 'vanilla' else args.option_selector.num_options

# words_dict = {i:[] for i in range(num_options)} # Create a list for each option
# num_eps = 1

# successes = []
    
# for i in range(num_eps):
#     for orig_instr, rephrasals in tqdm(LORL_EVAL_INSTRS.items()):
#         for rephrasal_type, instr_list in rephrasals.items():
#             for instr in instr_list:
#                 words, _, _, _, _, options_list, episode_length, success = traj(env, train_loader, model, tokenizer, args, render=False, orig_instr=orig_instr, instr=instr)
#                 #tokens = get_tokens(lm_inputs)
#                 tokens = instr.split()
#                 successes.append(success)
#                 for o in options_list:
#                     for w in tokens:
#                         words_dict[o].append(w)
# print(np.mean(successes))


# In[27]:


from tqdm import tqdm

num_options = 0 if args.method == 'vanilla' else args.option_selector.num_options
words_dict = {i:[] for i in range(num_options)} # Create a list for each option
num_eps = 5

successes = []
    
for i in range(num_eps):
    for orig_instr in tqdm(LORL_COMPOSITION_INSTRS):
        instr = orig_instr
        words, _, _, _, _, options_list, episode_length, success = traj(env, train_loader, model, tokenizer, args, render=False, orig_instr=orig_instr, instr=instr)
        #tokens = get_tokens(lm_inputs)
        tokens = instr.split()
        print(instr, options_list, success)
        successes.append(success)
        for o in options_list:
            for w in tokens:
                words_dict[o].append(w)
print(np.mean(successes))


# In[ ]:


# from tqdm import tqdm

# num_options = args.option_selector.num_options

# words_dict = {i:[] for i in range(num_options)} # Create a list for each option
# num_eps = 3
    
# for n in range(num_eps):
#     for i in tqdm(range(num_options)):
#         words, _, _, _, _, options_list, episode_length, success = traj(env, train_loader, model, tokenizer, args, render=True, orig_instr=f'{i}_{n+1}', instr=f'{i}_{n+1}', constant_option=i)
#         #tokens = get_tokens(lm_inputs)
#         tokens = instr.split()
#         #if success:
#         for o in options_list:
#             for w in tokens:
#                 words_dict[o].append(w)


# In[ ]:


# from itertools import chain

# skip_words = ['the', 'a', '[SEP]']

# words = sorted(set(chain(*words_dict.values())) - set(skip_words))
# print(words)


# # In[ ]:


# def w_to_ind(word):
#     return words.index(word)


# # In[ ]:


# matrix = np.zeros([len(words), num_options])

# for o in range(num_options):
#     for w in words_dict[o]:
#         if w not in skip_words:
#             matrix[w_to_ind(w), o] += 1
            
#print(matrix)


# In[ ]:


#!pip install seaborn


# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')
# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize = (30,10))
# sns.heatmap(matrix, yticklabels=words)
# plt.plot()


# # In[ ]:


# matrix.sum(axis=0, keepdims=True).shape


# # In[ ]:


# #get_ipython().run_line_magic('matplotlib', 'inline')
# # Now if we normalize it by column:
# plt.figure(figsize = (30,10))
# matrix_norm_col=(matrix)/(matrix.sum(axis=0, keepdims=True) + 1e-6)
# im = sns.heatmap(matrix_norm_col, yticklabels=words)
# plt.show()


# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')
# Now if we normalize it by row:
# plt.figure(figsize = (30,10))
# matrix_norm_row=(matrix)/(matrix.sum(axis=1, keepdims=True) + 1e-6)
# sns.heatmap(matrix_norm_row, yticklabels=words)
# plt.show()


# # In[ ]:


# #get_ipython().run_line_magic('matplotlib', 'inline')
# from collections import Counter
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud

# for i in range(num_options):
#     if words_dict[i]:
#         print(i)
#         cloud = WordCloud(max_font_size=80,colormap="hsv").generate_from_frequencies(Counter(x for x in words_dict[i] if x not in skip_words))
#         plt.figure(figsize=(16,12))
#         plt.imshow(cloud, interpolation='bilinear')
#         plt.axis('off')
#         plt.show()


# In[ ]:


#!pip install wordcloud


# In[ ]:





# In[ ]:





# In[ ]:





import torch
# import numpy as np

# import gym
# import babyai
# import lorl_env
# from env import BabyAIWrapper, LorlWrapper

# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.utils import set_random_seed

from viz import get_tokens


# def make_env(env, rank, seed=0):
#     """
#     Utility function for multiprocessed env.

#     :param env_id: (str) the environment ID
#     :param num_env: (int) the number of environments you wish to have in subprocesses
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     """

#     def _init():
#         env.seed(seed + rank)
#         return env

#     set_random_seed(seed)

#     return _init


# def evaluate():
#     env_id = 'BabyAI-BossLevel-v0'
#     #env_id = 'LorlEnv-v0'
#     num_cpu = 10  # Number of processes to use

#     env = SubprocVecEnv([make_env(BabyAIWrapper, env_id, i) for i in range(num_cpu)])
#     #env = SubprocVecEnv([make_env(LorlWrapper, env_id, i) for i in range(num_cpu)])
#     obs = env.reset()
#     print(obs.keys())
#     print(env.env_method('get_image'))
#     # env.render()

#     # for _ in range(1000):
#     #     action, _states = model.predict(obs)
#     #     obs, rewards, dones, info = env.step(action)

#     env.close()


# def parallel_eval_episode(
#         num_episodes, env, tokenizer, model, max_ep_len, K, words_dict, render, device, N=20, **kwargs):
#     """
#         Evaluate num_episodes episodes in parallel across N parallel envs
#     """
#     # keep track of total returns and episode_lengths
#     returns, lengths, success_list = [], [], []
#     episode_counts = np.zeros(N, dtype="int")
#     # Divides episodes among different sub environments in the vector as evenly as possible
#     episode_count_targets = np.array([(num_episodes + i) // N for i in range(N)], dtype="int")

#     env = SubprocVecEnv([make_env(env, i) for i in range(N)])
#     episode_return, episode_length, success = np.zeros(N), np.zeros(N), np.zeros(N)

#     states_list, actions_list, timesteps_list = [], [], []
#     images = []
#     method = model.method

#     horizon = model.horizon
#     if method != 'vanilla':
#         option_dim = model.option_selector.option_dim
#     else:
#         option_dim = None

#     if render and kwargs["i"] % kwargs["render_freq"] == 0:
#         observation = env.reset(
#             render=True, render_path=kwargs["render_path"],
#             iter_num=kwargs["iter_num"],
#             i=kwargs["i"])
#         cur_states, langs = observation["state"], observation["lang"]
#         images.append(env.env_method('get_image'))
#     else:
#         observation = env.reset(render=False)
#         cur_states, langs = observation["state"], observation["lang"]

#     state_dim = env.get_attr('state_dim', indices=0)[0]
#     act_dim = env.get_attr('act_dim', indices=0)[0]

#     if isinstance(state_dim, tuple):
#         cur_states = torch.from_numpy(cur_states).to(device=device, dtype=torch.float32).reshape(-1, *state_dim)
#     else:
#         cur_states = torch.from_numpy(cur_states).to(device=device, dtype=torch.float32).reshape(-1, state_dim)

#     states_list = [cur_states[i].reshape(1, state_dim) for i in range(N)]
#     actions_list = [torch.zeros((0, act_dim), device=device, dtype=torch.float32) for i in range(N)]
#     if method != 'vanilla':
#         options_list = [torch.zeros((0, option_dim), device=device, dtype=torch.float32) for i in range(N)]
#     else:
#         options_list = None
#     timesteps_list = [torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1) for i in range(N)]

#     # we keep all the histories on the device
#     # note that the latest action and reward will be "padding"
#     lm_input = tokenizer(text=langs, add_special_tokens=True, return_tensors='pt', padding=True).to(device=device)

#     with torch.no_grad():
#         lm_embeddings = model.lm(
#             lm_input['input_ids'], lm_input['attention_mask']).last_hidden_state
#         cls_embeddings = lm_embeddings[:, 0, :]
#         # word_embeddings = lm_embeddings[:, 1:-1, :]      # skip the CLS and SEP tokens.
#         word_embeddings = lm_embeddings[:, 1:, :]      # skip the CLS tokens

#     option = None
#     t = 0

#     while (episode_counts < episode_count_targets).any():
#         cur_actions = get_action_parallel(
#             model, states_list, actions_list, options_list, timesteps_list, cls_embeddings, word_embeddings,
#             options_list, cur_states, option, t, horizon, K, method, state_dim, act_dim, option_dim, device)
#         if model.discrete:
#             cur_actions = torch.nn.functional.one_hot(cur_actions, num_classes=act_dim)
#         else:
#             cur_actions = torch.clamp(cur_actions, torch.from_numpy(env.action_space.low).to(
#                 device), torch.from_numpy(env.action_space.high).to(device))
#         cur_actions = cur_actions.detach().cpu().numpy()

#         assert cur_actions in env.action_space, f"Transformer predicted action: {cur_actions} outside env action space {env.action_space}"

#         obs, rewards, dones, _ = env.step(cur_actions)

#         episode_return += rewards
#         episode_length += 1

#         # process state
#         cur_states = obs["state"]
#         if isinstance(state_dim, tuple):
#             cur_states = torch.from_numpy(cur_states).to(device=device, dtype=torch.float32).reshape(-1, *state_dim)
#         else:
#             cur_states = torch.from_numpy(cur_states).to(device=device, dtype=torch.float32).reshape(-1, state_dim)

#         for i in range(N):
#             if episode_counts[i] < episode_count_targets[i]:
#                 # terminate if done or reached env max_episode_length timesteps
#                 if dones[i] or states_list[i].shape[0] > max_ep_len:
#                     if episode_return[i] > 0:
#                         success[i] += 1

#                     returns.append(episode_return[i])
#                     lengths.append(episode_length[i])
#                     success_list.append(success[i])

#                     episode_return[i] = 0
#                     episode_length[i] = 0
#                     success[i] = 0
#                     # clear episode from list
#                     # TODO: THIS IS WRONG -- NEED TO CHANGE THE ENV TO RESET WHEN DONE BUT THIS IS A LITTLE UGLY
#                     states_list[i] = cur_states[i].reshape(1, state_dim)  # Use the new reset state
#                     actions_list[i] = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
#                     # Set initial timestep 0
#                     timesteps_list[i] = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

#                     # Only increment at the real end of an episode
#                     episode_counts[i] += 1

#         if render:
#             images.append(Image.fromarray(env.render(), 'RGB'))

#         states_list = [
#             torch.cat([states_list[i], cur_states[i].reshape(1, state_dim)], dim=0) for i in range(N)]
#         actions_list = [
#             torch.cat([actions_list[i], cur_actions[i].reshape(1, act_dim)], dim=0) for i in range(N)]
#         timesteps_list = [
#             torch.cat([timesteps_list[i], timesteps_list[i][-1].reshape(1, 1) + 1], dim=0) for i in range(N)]


def eval_episode(env, no_lang, tokenizer, model, max_ep_len, K, words_dict, render, device, **kwargs):
    """Evaluate a single episode."""
    images = []
    method = model.method

    horizon = model.horizon
    if method != 'vanilla':
        option_dim = model.option_selector.option_dim
    else:
        option_dim = None

    if render and kwargs["i"] % kwargs["render_freq"] == 0:
        observation = env.reset(
            render=True, render_path=kwargs["render_path"],
            iter_num=kwargs["iter_num"],
            i=kwargs["i"])
        cur_state, lang = observation['state'], observation['lang']
        images.append(env.get_image())
    else:
        observation = env.reset(render=False)
        cur_state, lang = observation["state"], observation["lang"]
    if no_lang:
        lang = ''

    state_dim = env.state_dim
    act_dim = env.act_dim
    cur_state = torch.from_numpy(cur_state)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    lm_input = tokenizer(text=[lang], add_special_tokens=True,
                         return_tensors='pt', padding=True).to(device=device)
    with torch.no_grad():
        lm_embeddings = model.lm(
            lm_input['input_ids'], lm_input['attention_mask']).last_hidden_state
        cls_embeddings = lm_embeddings[:, 0, :]
        # word_embeddings = lm_embeddings[:, 1:-1, :]      # skip the CLS and SEP tokens. here there's no padding so this is actually the CLS and SEP
        word_embeddings = lm_embeddings[:, 1:, :]      # skip the CLS tokens

    if isinstance(state_dim, tuple):
        states = cur_state.reshape(1, *state_dim).to(device=device, dtype=torch.float32)
    else:
        states = cur_state.reshape(1, state_dim).to(device=device, dtype=torch.float32)

    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    if method != 'vanilla':
        options = torch.zeros((0, option_dim), device=device, dtype=torch.float32)

    episode_return, episode_length, success = 0, 0, 0
    options_list = []

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
        if render and kwargs["i"] % kwargs["render_freq"] == 0:
            images.append(env.get_image())

        state, lang = obs['state'], obs['lang']
        if no_lang:
            lang = ''
            
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
            success = info.get('success', -1)
            break

    if method != 'vanilla' and model.option_selector.use_vq:
        tokens = get_tokens(lm_input, tokenizer)
        for o in options_list:
            for w in tokens:
                words_dict[o].append(w)

    return episode_return, episode_length, success, options_list, lang, images, words_dict


def get_action(
        model, states, actions, options, timesteps, cls_embeddings, word_embeddings, options_list, cur_state, option, t,
        horizon, K, method, state_dim, act_dim, option_dim, device, **kwargs):
    """
        Compute action for model evaluation
    """
    if method == 'vanilla':
        action = model.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            word_embeddings=word_embeddings.to(dtype=torch.float32)
        )
        return action, None, states, actions, timesteps, None
    else:
        if t % horizon == 0:
            # Choose a new option after every horizon steps
            if method == 'option':
                # option, option_index = model.option_selector.get_option(
                #     cls_embeddings, states[-1].reshape(1, 1, -1), **kwargs)
                option, option_index = model.option_selector.get_option(
                    word_embeddings.mean(1, keepdim=True), states[-1].reshape(1, 1, -1), **kwargs)
            else:
                option, option_index = model.option_selector.get_option(word_embeddings, states.to(dtype=torch.float32), timesteps.to(dtype=torch.long), **kwargs)
            options_list.append(option_index.cpu().item())

        if t % K == 0:
            # Reset state, actions, options and timesteps after every K steps
            actions = torch.zeros((1, act_dim), device=device, dtype=torch.float32)
            options = torch.zeros((1, option_dim), device=device, dtype=torch.float32)

            if isinstance(state_dim, tuple):
                states = cur_state.reshape(1, *state_dim).to(device=device, dtype=torch.float32)
            else:
                states = cur_state.reshape(1, state_dim).to(device=device, dtype=torch.float32)
            timesteps = torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)

        options[-1] = option

        action = model.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            options=options.to(dtype=torch.float32)
        )

        return action, option, states, actions, timesteps, options


# def get_action_parallel(
#         model, states, actions, options, timesteps, cls_embeddings, word_embeddings, options_list, cur_state, option, t,
#         horizon, K, method, state_dim, act_dim, option_dim, device):
#     pass


# if __name__ == '__main__':
#     evaluate()

from typing import Any, Dict, IO, List, Tuple

import numpy as np
import pandas as pd
import pickle
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from utils import pad, calculate_state_means_stds
from tqdm import tqdm


class ExpertDataset(Dataset):
    """Dataset for expert trajectories.
    Assumes expert dataset is a dict with keys {states, actions, rewards, lengths} with values
    of given shapes.
        shapes:
            expert["language"] = [num_experts]
            expert["states"] =  [num_experts, max_length, state_dim]
            expert["actions"] =  [num_experts, max_length, action_dim]
            expert["rewards"] =  [num_experts, max_length]
            expert["lengths"] =  [num_experts]
            expert["dones"] =  [num_experts, max_length]
    """

    def __init__(self,
                 expert_location: str,
                 num_trajectories: int = 1,
                 subsample_frequency: int = 1,
                 seed: int = 0,
                 full_traj: bool = True,
                 normalize_states: bool = True,
                 no_lang=False,
                 **kwargs):
        """Subsamples an expert dataset from saved expert trajectories.
        Args:
            expert_location:          Location of saved expert trajectories.
            num_trajectories:         Number of expert trajectories to sample (randomized).
            subsample_frequency:      Subsamples each trajectory at specified frequency of steps.
            seed:                     Seed for sampling trajectories.
            full_traj:                If True, each item will be a full trajectory and not just a (s,s',a,r,d) tuple
        """
        all_trajectories = load_trajectories(expert_location, num_trajectories, seed, **kwargs)
        self.kwargs = kwargs
        self.trajectories = {}
        self.full_traj = full_traj
        self.normalize_states = normalize_states
        self.no_lang = no_lang
        
        if 'aug' in self.kwargs and self.kwargs['aug']:
            ## From the LORL paper
            self.aug = torch.nn.Sequential(
                        transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.02),
                        transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(0.9, 1.1)))
      

        # skip the direction part for normalization
        if 'babyai' in expert_location and kwargs["use_direction"]:
            self.normalize_state_dim = all_trajectories["states"][0].shape[-1] - 4
        elif 'lorel' in expert_location:
            self.normalize_state_dim = all_trajectories["states"][0].shape[1:]
        elif 'calvin' in expert_location:
            self.normalize_state_dim = all_trajectories["states"][0].shape[1:]
        else:
            self.normalize_state_dim = np.array(all_trajectories["states"][0]).shape[1:]

        if normalize_states:
            self.state_mean, self.state_std = calculate_state_means_stds(
                all_trajectories["states"], self.normalize_state_dim)
        else:
            self.state_mean, self.state_std = np.zeros(self.normalize_state_dim), np.ones(self.normalize_state_dim)

        # Randomize start index of each trajectory for subsampling
        # start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories,)).long()

        # Subsample expert trajectories with every `subsample_frequency` step.
        for k, v in all_trajectories.items():
            data = v

            if k != "lengths":
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i][0::subsample_frequency])
                self.trajectories[k] = samples
            else:
                # Adjust the length of trajectory after subsampling
                self.trajectories[k] = np.array(data) // subsample_frequency

        if not full_traj:
            self.length = self.trajectories["lengths"].sum().item()
        else:
            self.length = len(self.trajectories["lengths"])
            self.total_timesteps = self.trajectories["lengths"].sum().item()

        del all_trajectories  # Not needed anymore

        self.max_length = max(self.trajectories["lengths"])

        # Convert flattened index i to trajectory indx and offset within trajectory
        if not self.full_traj:
            traj_idx = 0
            i = 0
            self.get_idx = []

            for _j in range(self.length):
                while self.trajectories["lengths"][traj_idx].item() <= i:
                    i -= self.trajectories["lengths"][traj_idx].item()
                    traj_idx += 1

                self.get_idx.append((traj_idx, i))
                i += 1

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, i):
        if not self.full_traj:
            traj_idx, i = self.get_idx[i]

            state = self.trajectories["states"][traj_idx][i]
            next_state = self.trajectories["states"][traj_idx][i+1]

            # Rescale states and next_states to [0, 1] if are images
            # if isinstance(states, np.ndarray) and states.ndim == 3:
            #     states = np.array(states) / 255.0
            # if isinstance(states, np.ndarray) and next_states.ndim == 3:
            #     next_states = np.array(next_states) / 255.0
            if self.normalize_states:
                state[:self.normalize_state_dim] = (
                    state[:self.normalize_state_dim] - self.state_mean) / self.state_std
                next_state[:self.normalize_state_dim] = (
                    next_state[:self.normalize_state_dim] - self.state_mean) / self.state_std

            return (state,
                    next_state,
                    self.trajectories["actions"][traj_idx][i],
                    self.trajectories["rewards"][traj_idx][i],
                    self.trajectories["dones"][traj_idx][i])
        else:
            states = self.trajectories["states"][i]
            states = np.array(states)
            if 'aug' in self.kwargs and self.kwargs['aug']:
                states = self.aug(torch.from_numpy(states))

            # Rescale states and next_states to [0, 1] if are images
            # if isinstance(states, np.ndarray) and states.ndim == 3:
            #     states = np.array(states) / 255.0
            if self.normalize_states:
                states[:, :self.normalize_state_dim] = (
                    states[:, :self.normalize_state_dim] - self.state_mean) / self.state_std

            timesteps = np.arange(0, states.shape[0])
            attention_mask = np.ones(states.shape[0])

            language = '' if self.no_lang else self.trajectories["language"][i]

            return (language,
                    pad(states, self.max_length, axis=0),
                    pad(self.trajectories["actions"][i], self.max_length, axis=0),
                    pad(timesteps, self.max_length, axis=0),
                    pad(self.trajectories["dones"][i], self.max_length, axis=0),
                    pad(attention_mask, self.max_length, axis=0)
                    )


def load_trajectories(expert_location: str,
                      num_trajectories: int = 10,
                      seed: int = 0,
                      **kwargs) -> Dict[str, Any]:
    """Load expert trajectories
    Args:
        expert_location:          Location of saved expert trajectories.
        num_trajectories:         Number of expert trajectories to sample (randomized).
        deterministic:            If true, random behavior is switched off.
    Returns:
        Dict containing keys {"states", "lengths"} and optionally {"actions", "rewards", "language", "dones"} with values
        containing corresponding expert data attributes.
    """
    if os.path.isfile(expert_location) or os.path.isdir(expert_location):
        # Load data from single file.
        if 'babyai' in expert_location:
            trajs = load_babyai_data(expert_location, num_trajectories, seed, **kwargs)
            # BabyAI does the random shuffling and taking subset for us
            return trajs
        elif 'lorel' in expert_location:
            trajs = load_lorel_data(expert_location, **kwargs)
        elif 'calvin' in expert_location:
            trajs = load_calvin_data(expert_location, num_trajectories, seed, **kwargs)
        else:
            with open(expert_location, 'rb') as f:
                trajs = read_file(expert_location, f)

            # Add empty strings if no language instructions in the trajectories
            if "language" not in trajs:
                trajs["language"] = ["" for i in range(len(trajs["states"]))]

        rng = np.random.RandomState(seed)
        # Sample random `num_trajectories` experts.
        perm = np.arange(len(trajs["states"]))
        perm = rng.permutation(perm)

        idx = perm[:num_trajectories]
        for k, v in trajs.items():
            # if not torch.is_tensor(v):
            #     v = np.array(v)  # convert to numpy array
            trajs[k] = [v[i] for i in idx]
    else:
        raise ValueError(f"{expert_location} is not a valid path")
    return trajs


def read_file(path: str, file_handle: IO[Any]) -> Dict[str, Any]:
    """Read file from the input path. Assumes the file stores dictionary data.
    Args:
        path:               Local or S3 file path.
        file_handle:        File handle for file.
    Returns:
        The dictionary representation of the file.
    """
    if path.endswith("pt"):
        data = torch.load(file_handle)
    elif path.endswith("pkl"):
        data = pickle.load(file_handle)
    elif path.endswith("npy"):
        data = np.load(file_handle, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()
    else:
        raise NotImplementedError
    return data


def load_babyai_data(expert_location, num_trajs, seed, **kwargs):
    from babyai.utils.demos import load_demos, transform_demos
    demos = transform_demos(load_demos(expert_location), num_trajs, seed)
    trajs = {"states": [], "actions": [], "rewards": [], "lengths": [], "language": [], "dones": []}
    for demo in tqdm(demos):
        trajs["language"].append(demo[0][0]["mission"])
        trajs["lengths"].append(len(demo))
        traj_states, traj_actions, traj_rewards, traj_dones = [], [], [], []
        for step in demo:
            obs, action, done = step
            if kwargs['use_direction']:
                direction = np.zeros(4)
                direction[obs["direction"]] = 1.
                state = np.concatenate([obs["image"].reshape(-1), direction])
            else:
                state = obs["image"].reshape(-1)
            traj_states.append(state)
            traj_actions.append(action)
            traj_rewards.append(None)
            traj_dones.append(done)
        trajs["states"].append(np.array(traj_states))
        trajs["actions"].append(np.array(traj_actions))
        trajs["rewards"].append(np.array(traj_rewards))
        trajs["dones"].append(np.array(traj_dones))

    assert len(trajs["states"]) == len(trajs["actions"]) == len(trajs["rewards"]) == len(
        trajs["lengths"]) == len(trajs["language"]) == len(trajs["dones"])
    return trajs


def load_lorel_data(expert_location, **kwargs):
    trajs = {"states": [], "actions": [], "rewards": [], "lengths": [], "language": [], "dones": []}
    data = pickle.load(open(expert_location, 'rb'))
    num_trajs, traj_len = data['actions'].shape[0], data['actions'].shape[1]
    if "state" in data.keys() and kwargs['use_state']:
        trajs["states"] = data["state"]
    else:
        trajs["states"] = np.moveaxis(data["ims"], 4, 2)  # making images C,H,W
    trajs["actions"] = data["actions"]
    trajs["rewards"] = np.array([[None for _ in range(traj_len)] for _ in range(num_trajs)])
    trajs["lengths"] = np.array([traj_len]*num_trajs)
    trajs["dones"] = np.array([[0]*(traj_len-1) + [1] for _ in range(num_trajs)])

    if 'sawyer' in expert_location:
        trajs["language"] = data['langs'].reshape(-1)
    elif 'franka' in expert_location:
        trajs["states"] = np.concatenate([trajs["states"], trajs["states"]], axis=0)
        trajs["actions"] = np.concatenate([trajs["actions"], trajs["actions"]], axis=0)
        trajs["rewards"] = np.concatenate([trajs["rewards"], trajs["rewards"]], axis=0)
        trajs["lengths"] = np.concatenate([trajs["lengths"], trajs["lengths"]], axis=0)
        trajs["dones"] = np.concatenate([trajs["dones"], trajs["dones"]], axis=0)
        trajs["language"] = data['langs'].T.reshape(-1)
    else:
        raise NotImplementedError

    assert len(
        trajs["states"]) == len(
        trajs["actions"]) == len(
        trajs["rewards"]) == len(
        trajs["lengths"]) == len(
        trajs["language"]) == len(
        trajs["dones"])
    return trajs


def load_calvin_data(expert_location, num_trajs, seed, **kwargs):
    trajs = {"states": [], "actions": [], "rewards": [], "lengths": [], "language": [], "dones": []}
    lang_anns = np.load(f'{expert_location}/lang_annotations/auto_lang_ann.npy', allow_pickle=True).item()
    np.random.seed(seed)
    chosen_traj_inds = np.random.choice(len(lang_anns['info']['indx']), size=num_trajs, replace=False)
    for i in chosen_traj_inds:
        start_idx, end_idx = lang_anns['info']['indx'][i]
        states, actions = [], []
        length = end_idx - start_idx + 1
        rewards = np.array([None] * length)
        language = lang_anns['language']['ann'][i]
        dones = np.array([0] * (length-1) + [1])
        for idx in range(start_idx, end_idx+1):
            info = np.load(f'{expert_location}/episode_{str(idx).zfill(7)}.npz')
            states.append(np.moveaxis(info['rgb_static'], 2,0))
            actions.append(info['actions'])
        assert len(states) == len(actions) == length
        states, actions = np.array(states), np.array(actions)
        trajs['states'].append(states)
        trajs['actions'].append(actions)
        trajs['rewards'].append(rewards)
        trajs['lengths'].append(length)
        trajs['language'].append(language)
        trajs['dones'].append(dones)
    assert len(
        trajs["states"]) == len(
        trajs["actions"]) == len(
        trajs["rewards"]) == len(
        trajs["lengths"]) == len(
        trajs["language"]) == len(
        trajs["dones"])
    return trajs


if __name__ == '__main__':
#     d = ExpertDataset('/atlas/u/divgarg/datasets/babyai/demos_iclr19_v2/BabyAI-GoToObj-v0_valid.pkl', 5, no_lang=True, normalize_states=False, use_direction=False)
#     print(d[0])
#     #print("**************")
#     #print(d[1])
#     del d

#     d = ExpertDataset('/atlas/u/divgarg/datasets/lorel/may_08_sawyer_50k/prep_data.pkl', 5, use_state=True)
#     print(d[0])
#     print("**************")
#     # print(d[1])
#     del d

#     d = ExpertDataset('/atlas/u/divgarg/datasets/lorel/may_08_sawyer_50k/prep_data.pkl', 5, use_state=False)
#     print(d[0])
#     print("**************")
#     print(d[1])
#     del d

#     d = ExpertDataset('/atlas/u/divgarg/datasets/lorel/may_06_franka_3k/prep_data.pkl', 5, use_state=False)
#     print(d[0])
#     print("**************")
#     # print(d[1])
#     del d

    d = ExpertDataset('/atlas/u/divgarg/datasets/calvin/dataset/task_D_D/training/', 5, normalize_states=False)
    print(d[0])
    print('****************')
    print(d[1])
    print('****************')
    del d
     

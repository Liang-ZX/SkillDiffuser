import numpy as np
import torch
import torch.nn.functional as F
import itertools
import gym
import h5py
import pandas as pd
import pickle
import cv2
import random
import string

import torch.distributions as pyd
from einops import rearrange, repeat
import math

# We addded these instructions
LORL_COMPOSITION_INSTRS = ['open drawer and move black mug right',
                           'pull the handle and move black mug down',
                           'move white mug right',
                           'move black mug down',
                           'close drawer and turn faucet right',
                           'close drawer and turn faucet left',
                           'turn faucet left and move white mug down',
                           'turn faucet right and close drawer',
                           'move white mug down and turn faucet left',
                           'close the drawer, turn the faucet left and move black mug right',
                           'open drawer and turn faucet counterclockwise',
                           'slide the drawer closed and then shift white mug down']

LORL_EVAL_INSTRS = {
    # From the paper
    'close drawer': {'seen': ['close drawer'],
                     'unseen verb': ['shut drawer'],
                     'unseen noun': ['close container'],
                     'unseen verb noun': ['shut container'],
                     'human': ['push the drawer shut',
                               'push the drawer',
                               'shut the drawer',
                               'shut drawer',
                               'slide the drawer closed',
                               'shut the drawer',
                               'shut the dresser',
                               'shut the drawer.',
                               'shut the cupboard']},
    'open drawer': {'seen': ['open drawer'],
                    'unseen verb': ['pull drawer'],
                    'unseen noun': ['open container'],
                    'unseen verb noun': ['pull container'],
                    'human': ['pull the drawer open',
                              'pull the handle',
                              'pull the drawer handle',
                              'pull the drawer open',
                              'pull open the drawer',
                              'open the dresser',
                              'pull the drawer',
                              'unclose the cabinet']},
    'turn faucet left': {'seen': ['turn faucet left'],
                         'unseen verb': ['rotate faucet left'],
                         'unseen noun': ['turn tap left'],
                         'unseen verb noun': ['rotate tap left'],
                         'human': ['rotate the tap counterclockwise',
                                   'turn faucet away from camera',
                                   'rotate nozzle left',
                                   'faucet counterclockwise',
                                   'rotate the faucet left',
                                   'turn the faucet to the left',
                                   'rotate handle to the left',
                                   'turn faucet counterclockwise',
                                   'spin nozzle left']},
    'turn faucet right': {'seen': ['turn faucet right'],
                          'unseen verb': ['rotate faucet right'],
                          'unseen noun': ['turn tap right'],
                          'unseen verb noun': ['rotate tap right'],
                          'human': ['rotate tap clockwise',
                                    'turn faucet towards camera',
                                    'rotate nozzle right',
                                    'faucet clockwise',
                                    'rotate the faucet right',
                                    'turn the faucet to the right',
                                    'rotate handle rightward',
                                    'turn faucet clockwise',
                                    'twirl valve right']},
    'move black mug right': {'seen': ['move black mug right'],
                             'unseen verb': ['push black mug right'],
                             'unseen noun': ['move dark cup right'],
                             'unseen verb noun': ['push dark cup right'],
                             'human': ['translate the black cup to the right',
                                       'move black mug away from drawer',
                                       'push black cup right',
                                       'black mug right',
                                       'slide the black mug right',
                                       'move the dark mug to the right',
                                       'push black cup right',
                                       'move black mug right.',
                                       'shift dark cup right']},
    'move white mug down': {'seen': ['move white mug down'],
                            'unseen verb': ['push white mug down'],
                            'unseen noun': ['move light cup down'],
                            'unseen verb noun': ['push light cup down'],
                            'human': ['translate the white cup down',
                                      'move white mug closer to the faucet',
                                      'bring white cup down',
                                      'white mug down',
                                      'push the white mug down and left',
                                      'move the lighter mug down',
                                      'shift white mug down',
                                      'pull white mug to the front.',
                                      'reposition white glass down']},
}


def lorl_save_im(im, name):
    """
        Save an image
    """
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, im.astype(np.uint8))


def lorl_gt_reward(qpos, initial, instr):
    """
        Measure true task progress for different instructions
    """
    if instr == "open drawer":
        dist = initial[14] - qpos[14]
        s = dist > 0.02
    elif instr == "close drawer":
        dist = qpos[14] - initial[14]
        s = dist > 0.02
    elif instr == "turn faucet right":
        dist = initial[13] - qpos[13]
        s = dist > np.pi / 10
    elif instr == "turn faucet left":
        dist = qpos[13] - initial[13]
        s = dist > np.pi / 10
    elif instr == "move black mug right":
        dist = initial[11] - qpos[11]
        s = dist > 0.02
    elif instr == "move white mug down":
        dist = qpos[10] - initial[10]
        s = dist > 0.02
    # We added these -- not sure if they are correct. Can add more combinations as well.
    elif instr == "open drawer and move black mug right":
        dist1 = initial[14] - qpos[14]
        dist2 = initial[11] - qpos[11]
        s = dist1 > 0.02 and dist2 > 0.02
        dist = (dist1 + dist2)/2
    elif instr == "move white mug right":
        dist = initial[10] - qpos[10]
        s = dist > 0.02
    elif instr == "move black mug down":
        dist = qpos[11] - initial[11]
        s = dist > 0.02
    elif instr == "close drawer and turn faucet right":
        dist1 = qpos[14] - initial[14]
        dist2 = initial[13] - qpos[13]
        s = dist1 > 0.02 and dist2 > np.pi/10
        dist = (dist1 + dist2)/2
    elif instr == "close drawer and turn faucet left":
        dist1 = qpos[14] - initial[14]
        dist2 = qpos[13] - initial[13]
        s = dist1 > 0.02 and dist2 > np.pi/10
        dist = (dist1+dist2)/2
    elif instr == "turn faucet left and move white mug down":
        dist1 = qpos[13] - initial[13]
        dist2 = qpos[10] - initial[10]
        s = dist1 > np.pi/10 and dist2 > 0.02
        dist = (dist1 + dist2)/2
    elif instr == "turn faucet right and close drawer":
        dist1 = initial[13] - qpos[13]
        dist2 = qpos[14] - initial[14]
        s = dist1 > np.pi/10 and dist2 > 0.02
        dist = (dist1+dist2)/2
    elif instr == "move white mug down and turn faucet left":
        dist1 = qpos[13] - initial[13]
        dist2 = qpos[10] - initial[10]
        s = dist1 > np.pi/10 and dist2 > 0.02
        dist = (dist1 + dist2)/2
    elif instr == "pull the handle and move black mug down":
        dist1 = initial[14] - qpos[14]
        dist2 = qpos[11] - initial[11]
        s = dist1 > 0.02 and dist2 > 0.02
        dist = (dist1 + dist2)/2
    elif instr == "close the drawer, turn the faucet left and move black mug right":
        dist1 = qpos[14] - initial[14]
        dist2 = qpos[13] - initial[13]
        dist3 = initial[11] - qpos[11]
        s = dist1 > 0.02 and dist2 > np.pi/10 and dist3 > 0.02
        dist = (dist1 + dist2 + dist3)/3
    elif instr == "open drawer and turn faucet counterclockwise":
        dist1 = initial[14] - qpos[14]
        dist2 = qpos[13] - initial[13]
        s = dist1 > 0.02 and dist2 > np.pi / 10
        dist = (dist1 + dist2)/2
    elif instr == "slide the drawer closed and then shift white mug down":
        dist1 = qpos[14] - initial[14]
        dist2 = qpos[10] - initial[10]
        s = dist1 > 0.02 and dist2 > 0.02
        dist = (dist1 + dist2)/2
    else:
        dist = 0
        s = 0
    return dist, s


class String(gym.Space):
    def __init__(
        self,
        length=None,
        min_length=0,
        max_length=512,
    ):
        self.length = length
        self.min_length = min_length
        self.max_length = max_length
        self.letters = string.ascii_letters + " .,!-"

    def sample(self):
        length = random.randint(self.min_length, self.max_length)
        string = ""
        for i in range(length):
            letter = random.choice(self.letters)
            string += letter
        return string

    def contains(self, x):
        return type(x) == "str" and len(x) > self.min and len(x) < self.max


def pad(x, max_len, axis=1, const=0, mode='pre'):
    """Pads input sequence with given const along a specified dim

    Inputs:
        x: Sequence to be padded
        max_len: Max padding length
        axis: Axis to pad (Default: 1)
        const: Constant to pad with (Default: 0)
        mode: ['pre', 'post'] Specifies whether to add padding pre or post to the sequence
    """

    if isinstance(x, tuple):
        x = np.array(x)

    pad_size = max_len - x.shape[axis]
    if pad_size <= 0:
        return x

    npad = [(0, 0)] * x.ndim
    if mode == 'pre':
        npad[axis] = (pad_size, 0)
    elif mode == 'post':
        npad[axis] = (0, pad_size)
    else:
        raise NotImplementedError

    if isinstance(x, np.ndarray):
        x_padded = np.pad(x, pad_width=npad, mode='constant', constant_values=const)
    elif isinstance(x, torch.Tensor):
        # pytorch starts padding from final dim so need to reverse chaining order
        npad = tuple(itertools.chain(*reversed(npad)))
        x_padded = F.pad(x, npad, mode='constant', value=const)
    else:
        raise NotImplementedError
    return x_padded


def calculate_state_means_stds(states_list, state_dim=None):
    # used for input normalization
    if state_dim is None:
        state_dim = states_list[0].shape[-1]
    states = [traj[:, :state_dim].reshape(-1, state_dim) for traj in states_list]
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    return state_mean, state_std


def preprocess_lorl_data():
    """
    Utility function to preprocess LORL data
    We filter the language instructions that are useless 
    """
    paths = ['/atlas/u/divgarg/datasets/lorel/may_08_sawyer_50k/', '/atlas/u/divgarg/datasets/lorel/may_06_franka_3k/']
    for path in paths:
        print(f'>>> Processing {path}')
        data = h5py.File(path + '/data.hdf5', 'r')['sim']
        labels = pd.read_csv(path + '/labels.csv')
        prep_data = {}
        if 'sawyer' in path:
            langs = labels["Text Description"].str.strip().to_numpy().reshape(-1)
            langs = np.array(['' if x is np.isnan else x for x in langs])
            filtr = np.array([int(("nothing" in l) or ("nan" in l) or ("wave" in l)) for l in langs])
            filtr = np.where(filtr == 0)[0]
            langs = langs[filtr]
        if 'franka' in path:
            langs1 = labels["Text Description 1"].str.strip().to_numpy().reshape(-1)
            langs2 = labels["Text Description 2"].str.strip().to_numpy().reshape(-1)
            langs1 = np.array(['' if x is np.isnan else x for x in langs1])
            langs2 = np.array(['' if x is np.isnan else x for x in langs2])
            filtr1 = np.array([int(("nothing" in l) or ("nan" in l) or ("wave" in l)) for l in langs1])
            filtr2 = np.array([int(("nothing" in l) or ("nan" in l) or ("wave" in l)) for l in langs2])
            filtr = np.where(filtr1 + filtr2 == 0)[0]
            langs1 = langs1[filtr].reshape(-1, 1)
            langs2 = langs2[filtr].reshape(-1, 1)
            langs = np.concatenate([langs1, langs2], -1)
        for key in data.keys():
            prep_data[key] = data[key][filtr, :, :]
            assert len(prep_data[key]) == len(langs)
        prep_data['langs'] = langs
        pickle.dump(prep_data, open(f'{path}/prep_data.pkl', 'wb'), protocol=4)
        print(f'>>> Preprocessed and stored {len(langs)} datapoints in {path}!')
        print(f'>>> Completed {path}!')

    print('All done!')


def entropy(codes, options, lang_state_embeds):
    """Calculate entropy of options over each batch

    option_codes: [N, D]
    lang_state_embeds: [B, D]
    """
    with torch.no_grad():
        N, D = codes.shape
        lang_state_embeds = lang_state_embeds.reshape(-1, 1, D)

        embed = codes.t()
        flatten = rearrange(lang_state_embeds, '... d -> (...) d')

        distance = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        # probs = (distance/2).exp() / math.sqrt(2 * math.pi)
        cond_probs = torch.softmax(distance / 2, dim=1)

        # dist = pyd.Independent(pyd.Normal(codes, torch.ones_like(codes)), 1)
        # probs = dist.log_prob(lang_state_embeds).exp()  # get probs as B x N

        # get marginal probabilities
        probs = cond_probs.mean(dim=0)

        entropy = (-torch.log2(probs) * probs).sum()

        # calculate conditional entropy with language
        # sum over options, and then take expectation over language
        cond_entropy = (-torch.log2(cond_probs) * cond_probs).sum(1).mean(0)
        return (entropy, cond_entropy)


if __name__ == '__main__':
    preprocess_lorl_data()

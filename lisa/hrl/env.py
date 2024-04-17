import numpy as np
import copy
import gym
from PIL import Image

from r3m import load_r3m
from utils import String, lorl_gt_reward, lorl_save_im

import torch
import torchvision.transforms as transforms
import metaworld
import cv2
from metaworld_utils import task_list
import random


class BaseWrapper(gym.Wrapper):
    """Base processing wrapper.

        1) Adds String command to observations
        2) Preprocess states
    """

    def __init__(self, env, dataset):
        super(BaseWrapper, self).__init__(env)
        self.env = env
        self.dataset = dataset
        self.state_mean, self.state_std = self.dataset.state_mean, self.dataset.state_std

        self.state_dim = len(self.state_mean)

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-np.inf,  # 0.0
                    high=np.inf,  # 1.0
                    shape=(self.state_dim,),
                    dtype=np.float32,
                ),
                "lang": String(),
            }
        )

        self.act_dim = env.action_space.shape[0]
        # print(self.action_space.low, self.action_space.high)

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self.get_state(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if done:
            success = 0
            if reward > 0:
                success = 1
            info.update({'success': success})

        return self.get_state(obs), reward, done, info

    def get_state(self, obs):
        """Returns the observation and lang_cmd"""

        lang = "dummy"
        cur_state = (obs.reshape(-1) - self.state_mean) / self.state_std

        return {'state': cur_state, 'lang': lang}

    def get_image(self):
        return Image.fromarray(self.env.render(), 'RGB')


class BabyAIWrapper(gym.Wrapper):
    """BabyAI processing wrapper.

        1) Adds String command to observations
        2) Preprocess states
    """

    def __init__(self, env, dataset):
        super(BabyAIWrapper, self).__init__(env)
        self.env = env
        self.dataset = dataset
        self.state_mean, self.state_std = self.dataset.state_mean, self.dataset.state_std

        self.state_dim = len(self.state_mean)
        if self.dataset.kwargs['use_direction']:
            self.state_dim += 4    # for the direction part in BabyAI

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-np.inf,  # 0.0
                    high=np.inf,  # 1.0
                    shape=(self.state_dim,),
                    dtype=np.float32,
                ),
                "lang": String(),
            }
        )

        self.act_dim = env.action_space.n

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self.get_state(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if done:
            success = 0
            if reward > 0:
                success = 1
            info.update({'success': success})

        return self.get_state(obs), reward, done, info

    def get_state(self, obs):
        """Returns the observation and lang_cmd"""

        lang = obs["mission"]
        cur_state = (obs["image"].reshape(-1) - self.state_mean) / self.state_std

        if self.dataset.kwargs['use_direction']:
            direction = np.zeros(4)
            direction[obs["direction"]] = 1.
            cur_state = np.concatenate([cur_state, direction])
        return {'state': cur_state, 'lang': lang}

    def get_image(self):
        return Image.fromarray(self.env.render(), 'RGB')


class MetaWorldWrapper(gym.Wrapper):
    """BabyAI processing wrapper.

        1) Adds String command to observations
        2) Preprocess states
    """

    def __init__(self, env, dataset, task, camera="corner", seed=0):
        env_name = task + '-v2'
        env_cls = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[
            env_name[:-3].replace('_', '-') + '-v2-goal-observable']
        env = env_cls(seed=seed)

        super(MetaWorldWrapper, self).__init__(env)
        self.env = env
        self.dataset = dataset
        self.env_name = task + '-v2'
        self.use_state = dataset.kwargs["use_state"]
        self.state_mean, self.state_std = self.dataset.state_mean, self.dataset.state_std

        # self.state_dim = env.observation_space.shape  # (39,)
        self.state_dim = 2048  # embedding dim
        self.act_dim = env.action_space.shape[0]

        self.res = (224, 224)
        self.camera = camera
        self.flip = False
        self.lang = task_list[task]

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.state_dim,),
                    dtype=np.float32,
                ),
                "lang": String(),
            }
        )

        self.action_space_ptp = env.action_space.high - env.action_space.low

        r3m = load_r3m("resnet50")  # resnet18, resnet34
        self.r3m = r3m

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.02),
            transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])

    def img2embed(self, img):
        pro_imgs = self.transform(Image.fromarray(img.astype(np.uint8))).reshape(-1, 3, 224, 224)

        device = "cuda"

        r3m = self.r3m
        r3m.eval()
        r3m.to(device)

        pro_imgs.to(device)
        with torch.no_grad():
            embedding = r3m(pro_imgs * 255.0)
        features = embedding.cpu().numpy()

        return features.squeeze()

    def reset(self, render=False, **kwargs):
        # if render:
        #     render_path, iter_num, i = kwargs['render_path'], kwargs['iter_num'], kwargs["i"]

        env = self.env

        env.reset()
        env.reset_model()

        last_o = env.reset()
        a = np.zeros(self.act_dim)
        o, _, _, info = env.step(a)

        for _ in range(100):
            self.env.sim.step()

        cur_state = o
        lang = self.lang

        if self.use_state:
            cur_state = (cur_state - self.state_mean) / self.state_std
            self.state_dim = len(cur_state)
        else:
            im = env.sim.render(*self.res, mode='offscreen', camera_name=self.camera)[:, :, ::-1]
            # im = np.moveaxis(im, 2, 0)  # make H,W,C to C,H,w
            # cur_state = (im - self.state_mean) / self.state_std
            cur_state = self.img2embed(im)
            self.state_dim = cur_state.shape

        return {'state': cur_state, 'lang': lang}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # if done:
        #     success = 0
        #     if reward > 0:
        #         success = 1
        #     info.update({'success': success})

        return self.get_state(obs), reward, done, info

    def get_state(self, obs):
        """Returns the observation and lang_cmd"""

        if self.use_state:
            obs = obs
            state = (obs - self.state_mean) / self.state_std
        else:
            im = self.env.sim.render(*self.res, mode='offscreen', camera_name=self.camera)[:, :, ::-1]
            # obs = np.moveaxis(im, 2, 0)  # make H,W,C to C,H,w
            state = self.img2embed(im)

        return {'state': state, 'lang': self.lang}

    def get_image(self, h=1024, w=1024):
        im = self.env.sim.render(h, w, mode='offscreen', camera_name=self.camera)[:, :, ::-1]
        if self.flip: im = cv2.rotate(im, cv2.ROTATE_180)

        return im.astype(np.uint8)


class LorlWrapper(gym.Wrapper):
    """BabyAI processing wrapper.

        1) Adds String command to observations
        2) Preprocess states
    """

    def __init__(self, env, dataset, **kwargs):
        super(LorlWrapper, self).__init__(env)

        self.env = env
        self.dataset = dataset
        self.use_state = dataset.kwargs["use_state"]
        self.state_mean, self.state_std = self.dataset.state_mean, self.dataset.state_std

        self.state_dim = self.state_mean.shape
        self.act_dim = env.action_space.shape[0]

        if isinstance(self.state_dim, tuple):
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=self.state_dim,
                        dtype=np.float32,
                    ),
                    "lang": String(),
                }
            )
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.state_dim,),
                        dtype=np.float32,
                    ),
                    "lang": String(),
                }
            )

        self.initial_state = None
        self.instr = kwargs["instr"]
        self.orig_instr = kwargs["orig_instr"]

    def reset(self, render=False, **kwargs):
        if render:
            render_path, iter_num, i = kwargs['render_path'], kwargs['iter_num'], kwargs["i"]

        env = self.env
        orig_instr, instr = self.orig_instr, self.instr
        im, _ = env.reset()

        # Initialize state for different tasks
        if orig_instr == "open drawer":
            env.sim.data.qpos[14] = 0 + np.random.uniform(-0.05, 0)
        elif orig_instr == "close drawer":
            env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == "turn faucet right":
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
        elif orig_instr == "turn faucet left":
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
        elif orig_instr == "move black mug right":
            env.sim.data.qpos[11] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[12] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == "move white mug down":
            env.sim.data.qpos[9] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[10] = 0.65 + np.random.uniform(-0.05, 0.05)
        # Dont know if the following are correct
        elif orig_instr == 'open drawer and move black mug right':
            env.sim.data.qpos[14] = 0 + np.random.uniform(-0.05, 0)
            env.sim.data.qpos[11] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[12] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'pull the handle and move black mug down':
            env.sim.data.qpos[14] = 0 + np.random.uniform(-0.05, 0)
            env.sim.data.qpos[11] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[12] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'move white mug right':
            env.sim.data.qpos[9] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[10] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'move black mug down':
            env.sim.data.qpos[11] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[12] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'close drawer and turn faucet right':
            env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
        elif orig_instr == 'close drawer and turn faucet left':
            env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
        elif orig_instr == 'turn faucet left and move white mug down':
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
            env.sim.data.qpos[9] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[10] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'turn faucet right and close drawer':
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
            env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'move white mug down and turn faucet left':
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
            env.sim.data.qpos[9] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[10] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif orig_instr == 'close the drawer, turn the faucet left and move black mug right':
            env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
            env.sim.data.qpos[11] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[12] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif instr == "open drawer and turn faucet counterclockwise":
            env.sim.data.qpos[14] = 0 + np.random.uniform(-0.05, 0)
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
        elif instr == "slide the drawer closed and then shift white mug down":
            env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[9] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[10] = 0.65 + np.random.uniform(-0.05, 0.05)

        # if orig_instr == "move white mug down":
        #    env._reset_hand(pos=[-0.1, 0.55, 0.1])
        # elif orig_instr == "move black mug right":
        #    env._reset_hand(pos=[-0.1, 0.55, 0.1])
        if "mug" in orig_instr:
            env._reset_hand(pos=[-0.1, 0.55, 0.1])
        else:
            env._reset_hand(pos=[0, 0.45, 0.1])

        for _ in range(50):
            env.sim.step()

        reset_state = copy.deepcopy(env.sim.data.qpos[:])
        env.sim.data.qpos[:] = reset_state
        env.sim.data.qacc[:] = 0
        env.sim.data.qvel[:] = 0
        env.sim.step()
        self.initial_state = copy.deepcopy(env.sim.data.qpos[:])

        if render:
            # Initialize goal image for initial state
            if orig_instr == "open drawer":
                env.sim.data.qpos[14] = -0.15
            elif orig_instr == "close drawer":
                env.sim.data.qpos[14] = 0.0
            elif orig_instr == "turn faucet right":
                env.sim.data.qpos[13] -= np.pi/5
            elif orig_instr == "turn faucet left":
                env.sim.data.qpos[13] += np.pi/5
            elif orig_instr == "move black mug right":
                env.sim.data.qpos[11] -= 0.1
            elif orig_instr == "move white mug down":
                env.sim.data.qpos[10] += 0.1

            env.sim.step()
            gim = env._get_obs()[:, :, :3]

            # Reset inital state
            env.sim.data.qpos[:] = reset_state
            env.sim.data.qacc[:] = 0
            env.sim.data.qvel[:] = 0
            env.sim.step()

            im = env._get_obs()[:, :, :3]
            initim = im
            lorl_save_im(
                (initim * 255.0).astype(np.uint8),
                render_path + f"/initialim_{iter_num}_{i}_{instr}.jpg")
            lorl_save_im((gim*255.0).astype(np.uint8), render_path+f"gim_{iter_num}_{i}_{instr}.jpg")

        observation = self.get_state(im)
        cur_state, lang = observation['state'], observation['lang']
        if self.use_state:
            cur_state = (cur_state - self.state_mean) / self.state_std
            self.state_dim = len(cur_state)
        else:
            im = np.moveaxis(im, 2, 0)  # make H,W,C to C,H,w
            cur_state = (im - self.state_mean) / self.state_std
            self.state_dim = cur_state.shape

        return {'state': cur_state, 'lang': lang}

    def step(self, action):
        im, _, _, info = self.env.step(action)
        dist, s = lorl_gt_reward(self.env.sim.data.qpos[:], self.initial_state, self.orig_instr)

        reward = 0
        success = 0
        if s:
            success = 1
            reward = dist

        info.update({'success': success})
        return self.get_state(im), reward, s, info

    def get_state(self, obs):
        """Returns the observation and lang_cmd"""

        if self.use_state:
            obs = self.env.sim.data.qpos[:]
        else:
            obs = np.moveaxis(obs, 2, 0)  # make H,W,C to C,H,w

        state = (obs - self.state_mean) / self.state_std
        return {'state': state, 'lang': self.instr}

    def get_image(self, h=1024, w=1024):
#         im = self.env._get_obs()
        obs = self.sim.render(h, w, camera_name="cam0") / 255.
        im = np.flip(obs, 0).copy()
        return (im[:, :, :3]*255.0).astype(np.uint8)

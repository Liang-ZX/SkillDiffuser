import numpy as np
import copy
import gym
from PIL import Image
import pdb

from utils import String, lorl_gt_reward, lorl_save_im


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
            render_path = "."
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

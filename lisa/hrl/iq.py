import numpy as np
from numpy.core.shape_base import vstack
import torch
from torch._C import qscheme
import torch.nn.functional as F
from torch.distributions import Categorical
import copy
import wandb


class IQMixin(object):
    # Mixin to Base model that adds extra IQ

    def __init__(self, q_net, args, device):
        super().__init__()

        self.gamma = args.gamma
        self.args = args
        self.device = device

        self.log_alpha = torch.tensor(np.log(args.alpha)).to(self.device)
        self.q_net = q_net

        self.train()

        # Create target network
        if args.use_target:
            self.target_net = copy.deepcopy(q_net)
            self.target_net.load_state_dict(self.q_net.state_dict())

            self.target_net.train()
        # self.critic_tau = agent_cfg.critic_tau
        # self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency

    def train(self, training=True):
        self.training = training
        self.q_net.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.q_net

    @property
    def critic_target_net(self):
        return self.target_net

    def iq_choose_action(self, q, sample=True):
        with torch.no_grad():
            dist = F.softmax(q/self.alpha, dim=1)
            if sample:
                dist = Categorical(dist)
                action = dist.sample()  # if sample else dist.mean
            else:
                action = torch.argmax(dist, dim=1)
        return action
        # return action.detach().cpu().numpy()[0]

    def getV(self, q):
        v = self.alpha * \
            torch.logsumexp(q/self.alpha, dim=1, keepdim=True)
        return v

    def critic(self, q, action):
        return q.gather(1, action.long())

    # Offline IQ-Learn objective
    def iq_update_critic(self, expert_batch):
        args = self.args
        # Assume the expert_batch contains the current state q and the next state q.
        q, next_q, action, done = expert_batch

        losses = {}
        # keep track of v0
        v0 = self.getV(q).mean()
        losses['v0'] = v0.item()

        # calculate 1st term of loss
        #  -E_(ρ_expert)[Q(s, a) - γV(s')]
        current_q = self.critic(q, action)
        next_v = self.getV(next_q)
        y = (1 - done) * self.gamma * next_v

        if args.use_target:
            with torch.no_grad():
                target_q = self.target_net(next_obs)
                next_v = self.get_V(target_q)
                y = (1 - done) * self.gamma * next_v

        reward = current_q - y

        with torch.no_grad():
            if args.div == "hellinger":
                phi_grad = 1/(1+reward)**2
            elif args.div == "kl":
                phi_grad = torch.exp(-reward-1)
            elif args.div == "kl2":
                phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
            elif args.div == "kl_fix":
                phi_grad = torch.exp(-reward)
            elif args.div == "js":
                phi_grad = torch.exp(-reward)/(2 - torch.exp(-reward))
            else:
                phi_grad = 1

        loss = -(phi_grad * reward).mean()
        losses['softq_loss'] = loss.item()

        if args.loss == "v0":
            # calculate 2nd term for our loss
            # (1-γ)E_(ρ0)[V(s0)]
            v0_loss = (1 - self.gamma) * v0
            loss += v0_loss
            losses['v0_loss'] = v0_loss.item()

        elif args.loss == "value":
            # alternative 2nd term for our loss (use only expert states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (self.getV(q) - y).mean()
            loss += value_loss
            losses['value_loss'] = value_loss.item()

        elif args.loss == "skip":
            # No loss
            pass

        if args.div == "chi":
            # Use χ2 divergence (adds a extra term to the loss)
            if args.use_target:
                with torch.no_grad():
                    target_q = self.target_net(next_obs)
                    next_v = self.getV(target_q)
            else:
                next_v = self.getV(next_q)

            y = (1 - done) * self.gamma * next_v

            current_q = self.critic(q, action)
            reward = current_q - y
            chi2_loss = 1/2 * (reward**2).mean()
            loss += chi2_loss
            losses['chi2_loss'] = chi2_loss.item()

        losses['total_loss'] = loss.item()

        return loss, losses

    def iq_critic_loss(self, batch, step):
        # assume we get input of trajectory with state, action, q_predictions, dones
        q, next_q, actions, dones = batch

        actions = actions.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        loss, loss_metrics = self.iq_update_critic((q, next_q, actions, dones))

        if self.args.use_target and step % self.critic_target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        wandb.log(loss_metrics, step=step)
        return loss

    def infer_q(self, state, action):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = torch.FloatTensor([action]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.critic(state, action)
        return q.squeeze(0).cpu().numpy()

    def infer_v(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            v = self.getV(state).squeeze()
        return v.cpu().numpy()

# import pdb

import numpy as np
import torch
import copy
from diffuser.utils.training import EMA
from ml_logger import logger
import os
import torch.nn as nn

class DiffTrainer(nn.Module):
    def __init__(
            self,
            args,
            diffusion_model,
            diff_trainer_args,
            ):
        # max_length used to be K
        super().__init__()

        self.model = diffusion_model
        self.ema = EMA(diff_trainer_args['ema_decay'])
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = diff_trainer_args['update_ema_every']
        self.save_checkpoints = diff_trainer_args['save_checkpoints']
        self.step_start_ema = 2000

        self.log_freq = diff_trainer_args['log_freq']
        self.sample_freq = diff_trainer_args['sample_freq']
        self.save_freq = diff_trainer_args['save_freq']
        self.label_freq = diff_trainer_args['label_freq']
        self.save_parallel = diff_trainer_args['save_parallel']

        # self.batch_size = diff_trainer_args['train_batch_size']
        self.gradient_accumulate_every = diff_trainer_args['gradient_accumulate_every']

        self.bucket = os.path.join(args.hydra_base_dir, "buckets")
        self.n_reference = diff_trainer_args['n_reference']

        self.reset_parameters()
        self.step = 0

        self.device = diff_trainer_args['train_device']

        logger.configure(args.hydra_base_dir,
                         prefix=f"diffuser_log")
        # self.model = self.model.to(device=self.device)  # TODO
        # self.ema_model = self.ema_model.to(device=self.device)

        self.lr = diff_trainer_args['train_lr']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train_iteration(self, batch):
        # timer = Timer()
        # for step in range(n_train_steps):
        # for i in range(self.gradient_accumulate_every):
            # batch = next(self.dataloader)
            # batch = batch_to_device(batch, device=self.device)
        loss, infos = self.model.loss(*batch)
        # loss = loss / self.gradient_accumulate_every

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        if self.step % self.update_ema_every == 0:
            self.step_ema()

        if self.step % self.save_freq == 0:
            self.save()

        infos.pop('obs')

        if self.step % self.log_freq == 0:
            infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
            logger.print(f'{self.step}: {loss:8.4f} | {infos_str}')
            metrics = {k: v.detach().item() for k, v in infos.items()}
            metrics['steps'] = self.step
            metrics['loss'] = loss.detach().item()
            logger.log_metrics_summary(metrics, default_stats='mean')

        # if self.step == 0 and self.sample_freq:
        #     self.render_reference(self.n_reference)

        # if self.sample_freq and self.step % self.sample_freq == 0:
        #     if self.model.__class__ == diffuser.models.diffusion.GaussianInvDynDiffusion:
        #         self.inv_render_samples()
        #     elif self.model.__class__ == diffuser.models.diffusion.ActionGaussianDiffusion:
        #         pass
        #     else:
        #         self.render_samples()

        self.step += 1
        return

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.bucket, logger.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def load(self, loadpath):
        '''
            loads model and ema from disk
        '''
        # loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        # self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

import pdb

import torch
import torch.nn as nn
from transformers import DistilBertModel

from iq import IQMixin
from option_selector import OptionSelector
from reconstructors import StateReconstructor, LanguageReconstructor
# from decision_transformer import DecisionTransformer
from dec_encoder import DecEncoder
# from utils import pad
# import diffuser.utils as diffutils


class HRLModel(nn.Module, IQMixin):
    """Base class containing all the models"""

    def __init__(self, args, option_selector_args, state_reconstructor_args, lang_reconstructor_args,
                 decision_args, iq_args, diff_trainer, device, horizon=5, K=10, train_lm=True,
                 method='vanilla', state_reconstruct=False, lang_reconstruct=False, **kwargs):
        super().__init__()

        self.args = args
        self.horizon = horizon
        self.K = K
        self.lm = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.train_lm = train_lm  # whether to train lm or not
        self.device = device

        if train_lm:
            self.lm.train()
        else:
            self.lm.eval()

        self.method = method
        self.state_reconstruct = state_reconstruct
        self.lang_reconstruct = lang_reconstruct

        self.state_dim = decision_args['state_dim']
        self.action_dim = decision_args['action_dim']
        self.option_dim = decision_args['option_dim']

        # self.decision_transformer = DecisionTransformer(lang_dim=self.lm.config.dim, **decision_transformer_args)

        if method == 'vanilla':
            assert decision_args['use_language'] and not decision_args['use_options']
        else:
            assert decision_args['use_options'] and not decision_args['use_language']

            self.option_selector = OptionSelector(lang_dim=self.lm.config.dim,
                                                  method=self.method, **option_selector_args)

            if state_reconstruct:
                self.state_reconstructor = StateReconstructor(**state_reconstructor_args)
            if lang_reconstruct:
                self.lang_reconstructor = LanguageReconstructor(
                    lang_dim=self.lm.config.dim, **lang_reconstructor_args)

        # TODO
        self.diffuser = DecEncoder(lang_dim=self.lm.config.dim, K=K, **decision_args)

        self.diff_trainer = diff_trainer
        # self.diff_trainer.set_optimizer(self.diffuser)

        # if self.decision_transformer.predict_q:
        #     # initialize iq mixins
        #     IQMixin.__init__(self, self.decision_transformer, iq_args, device)

    def forward(self, lm_input_ids, lm_attention_mask, states, actions, timesteps, iter_num, ind, diff_train_num, attention_mask=None):
        # pdb.set_trace()
        batch_size, traj_len = states.shape[0], states.shape[1]
        if not self.train_lm:
            with torch.no_grad():
                # (batch_size,num_embeddings,embedding_size)
                lm_embeddings = self.lm(lm_input_ids, lm_attention_mask).last_hidden_state
        else:
            # (batch_size,num_embeddings,embedding_size)
            lm_embeddings = self.lm(lm_input_ids, lm_attention_mask).last_hidden_state
        cls_embeddings = lm_embeddings[:, 0, :].unsqueeze_(1)
        # word_embeddings = lm_embeddings[:, 1:-1, :]      # We skip the CLS and SEP tokens. I know there's padding here but we at least always remove the CLS
        word_embeddings = lm_embeddings[:, 1:, :]      # We skip the CLS tokens

        entropy = None
        if self.method == 'vanilla':
            raise NotImplementedError
            # preds = self.decision_transformer(
            #     states, actions, timesteps, word_embeddings=word_embeddings, attention_mask=attention_mask)
            #
            # state_rc_preds = None
            # state_rc_targets = None
            #
            # lang_rc_preds = None
            # lang_rc_targets = None
            #
            # commitment_loss = None
        else:
            # TODO (important) how does this get padded across batches? some of these horizon states may actually be padding
            # we change options only every H states. Say this leads to N states
            # N selected options
            # B, max_length // H, option_dim
            if self.method == 'option':
                # selected_options, _, commitment_loss = self.option_selector(cls_embeddings, states)
                selected_options, _, commitment_loss, entropy, state_embeddings = self.option_selector(
                    word_embeddings.mean(1, keepdim=True), states)
            else:
                selected_options, _, commitment_loss, entropy, state_embeddings = self.option_selector(
                    word_embeddings, states, timesteps, attention_mask)

            # # need to make options same length as states and actions
            options = torch.zeros((batch_size, traj_len, selected_options.shape[-1])).to(selected_options.device)

            ### This doesn't really work in making only some messages have gradients and others not having gradients
            ### The entire options tensor below has gradients after we do options = selected_options
            ### Actually it may only have the gradients related to the selectbackward operation -- unsure

            # Repeated detached options for horizon length
            for i in range(selected_options.shape[1]):
                options[:, i*self.horizon:(i+1)*self.horizon, :] = selected_options[:,
                                                                                    i, :].unsqueeze(1).clone().detach()
            # Make sure to pass gradients for options at each horizon steps
            options[:, ::self.horizon, :] = selected_options

            # We reshape sequences to K size sub-sequences, so that the sub-policy only uses the current option
            # Here we are choosing K to be horizon
            B, L = states.shape[0], states.shape[1]
            num_seq = L // self.K  # self.K == self.horizon == 8

            # We reshape sequences to K size sub-sequences, so that the sub-policy only uses the current option
            # Here we are choosing K to be horizon since it makes sense but technically we can do any K
            # This ensures the DT only looks at chunks of size horizon
            if isinstance(self.state_dim, tuple):
                states = states.reshape(B * num_seq, self.K, *self.state_dim)
            else:
                states = states.reshape(B * num_seq, self.K, self.state_dim)
            state_embeddings = state_embeddings.reshape(B * num_seq, self.K, state_embeddings.shape[2])
            options = options.reshape(B * num_seq, self.K, self.option_dim)
            actions = actions.reshape(B * num_seq, self.K, self.action_dim)
            # Should these timesteps be 1,2,3,4..H,1,2... or just 1,2,3,4...L? Going with 1,2,3,4...L
            timesteps = timesteps.reshape(B * num_seq, self.K)
            # timesteps = torch.arange(0, self.K).repeat(B * num_seq, 1)
            attention_mask = attention_mask.reshape(B * num_seq, self.K)

            # Make sure shapes are okay
            assert states.shape[0] == actions.shape[0] == options.shape[0] == timesteps.shape[0] == attention_mask.shape[0] == batch_size * num_seq
            assert states.shape[1] == actions.shape[1] == options.shape[1] == timesteps.shape[1] == attention_mask.shape[1] == self.K
            # preds = self.decision_transformer(
            #     states, actions, timesteps, options=options, attention_mask=attention_mask)

            # TODO
            encoder_out = self.diffuser.encode(states, actions, timesteps, options=options,
                                               state_embeddings=state_embeddings, attention_mask=attention_mask)

            stacked_inputs = encoder_out['stacked_inputs'].to(self.device)
            option_embeddings = encoder_out['option_embeddings'].to(self.device)
            conds = {}
            returns = option_embeddings  # conditions
            # num_diff_steps = int(self.args.diffuser.n_train_steps // self.args.max_iters)
            for in_step in range(diff_train_num):
                # step = iter_num * num_diff_steps + ind * diff_train_num + in_step
                self.diff_trainer.train_iteration((stacked_inputs, conds, returns))

            diff_loss, infos = self.diff_trainer.model.loss(stacked_inputs, conds, returns)
            obs_recon = infos.pop('obs')

            decode_outputs = self.diffuser.decode(obs_recon)

            self.state_reconstruct = False
            self.lang_reconstruct = False
            if self.state_reconstruct:
                # TODO: Maybe fix?? We now predict an option using trajs
                state_rc_preds = self.state_reconstructor(selected_options)
                state_rc_targets = states  # horizon_states
            else:
                state_rc_preds = None
                state_rc_targets = None

            if self.lang_reconstruct:
                # TODO: Maybe fix?? Do we need the max options formulation? Check this
                lang_rc_preds = self.lang_reconstructor(selected_options.reshape(batch_size, -1))
                lang_rc_targets = cls_embeddings
            else:
                lang_rc_preds = None
                lang_rc_targets = None

        return {'dt': decode_outputs,
                'state_rc': (state_rc_preds, state_rc_targets),
                'lang_rc': (lang_rc_preds, lang_rc_targets),
                'actions': actions,
                'attention_mask': attention_mask,
                'commitment_loss': commitment_loss,
                'diff_loss': diff_loss,
                'entropy': entropy}

    def get_action(self, states, actions, timesteps, options=None, word_embeddings=None, ema_diffusion_model=None):
        if self.method == 'vanilla':
            preds = self.decision_transformer.get_action(
                states, actions, timesteps, word_embeddings=word_embeddings)

            return preds
        else:
            # preds = self.decision_transformer.get_action(
            #     states, actions, timesteps, options=options)
            encoder_out = self.diffuser.get_action(states, actions, timesteps,
                                                   options=options, embed_state=self.option_selector.option_dt.embed_state)
            stacked_inputs = encoder_out['stacked_inputs']
            option_embeddings = encoder_out['option_embeddings']

            conds = {0: stacked_inputs[:, -1, self.diffuser.act_dim:]}
            returns = option_embeddings  # conditions

            samples = ema_diffusion_model.conditional_sample(conds, returns=returns)

            action_hist = []
            for tt in range(samples.shape[1]-1):
                obs_comb = torch.cat([samples[:, tt, :], samples[:, tt+1, :]], dim=-1)
                obs_comb = obs_comb.reshape(-1, 2 * self.diffuser.hidden_size)
                action = ema_diffusion_model.inv_model(obs_comb)
                action_hist.append(action)

            # obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            # obs_comb = obs_comb.reshape(-1, 2 * self.diffuser.hidden_size)
            # action = ema_diffusion_model.inv_model(obs_comb)

        if self.diffuser.predict_q:
            # Choose actions from q_values
            # action = self.iq_choose_action(preds['q_preds'][:, -1], sample=True)
            raise NotImplementedError
        else:
            # Choose actions from direct predictions
            # action = preds['action_preds'][:, -1]
            if self.diffuser.discrete:
                action = action.argmax(dim=1)

        # action = action.squeeze(0)
        action = torch.cat(action_hist, dim=0)
        return action

    def save(self, iter_num, filepath, config):
        if hasattr(self.model, 'module'):
            model = self.model.module

        torch.save({'model': model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'iter_num': iter_num,
                    'train_dataset_max_length': self.train_loader.dataset.max_length,
                    'config': config}, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return {'iter_num': checkpoint['iter_num'], 'train_dataset_max_length': checkpoint['train_dataset_max_length'], 'config': checkpoint['config']}

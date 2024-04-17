import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from trajectory_model import TrajectoryModel
from trajectory_gpt2 import GPT2Model
from img_encoder import Encoder

import iq
from utils import pad


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Lang, state_1, action_1, state_2, ...) or (state_1, option_1, action_1, ...)
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            option_dim,
            lang_dim,
            discrete,
            hidden_size,
            use_language=False,
            use_options=True,
            option_il=False,
            predict_q=False,
            max_length=None,
            max_ep_len=4096,
            action_tanh=False,
            no_states=False,
            no_actions=False,
            ** kwargs):
        # max_length used to be K
        super().__init__(state_dim, action_dim, max_length=max_length)

        self.use_options = use_options
        self.use_language = use_language
        self.option_il = option_il
        self.predict_q = predict_q

        if use_language and use_options:
            raise ValueError("Cannot use language and options!")
        if not use_language and not use_options:
            raise ValueError("Have to use language or options!")
        self.option_dim = option_dim
        self.discrete = discrete

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        if isinstance(state_dim, tuple):
            # LORL
            if state_dim[0] == 3:
                # LORL Sawyer
                self.embed_state = Encoder(hidden_size=hidden_size, ch=3, robot=False)
            else:
                # LORL Franka
                self.embed_state = Encoder(hidden_size=hidden_size, ch=12, robot=True)
        else:
            self.embed_state = nn.Linear(self.state_dim, hidden_size)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        self.embed_action = nn.Linear(self.act_dim, hidden_size)

        self.no_states = no_states
        self.no_actions = no_actions

        if use_options:
            self.embed_option = nn.Linear(self.option_dim, hidden_size)

        if use_language:
            self.embed_lang = nn.Linear(lang_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        # note: we don't predict states or returns for the paper
        if isinstance(self.state_dim, int):
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh and not discrete else []))
        )
        if use_options:
            self.predict_option = torch.nn.Linear(hidden_size, self.option_dim)
        if predict_q:
            self.predict_q = torch.nn.Linear(hidden_size, self.act_dim)

    def forward(self, states, actions, timesteps, options=None, word_embeddings=None, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            raise ValueError('Should not have attention_mask NONE')
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        if self.use_options:
            assert options is not None
            option_embeddings = self.embed_option(options)
            time_embeddings = self.embed_timestep(timesteps)

            # time embeddings are treated similar to positional embeddings
            option_embeddings = option_embeddings + time_embeddings

            if self.no_states:
                # IMP: MAKE SURE THIS IS NOT SET ON BY DEFAULT
                state_embeddings = self.embed_state(torch.zeros_like(states))
            else:
                state_embeddings = self.embed_state(states)
                state_embeddings = state_embeddings + time_embeddings

            if self.no_actions:
                # IMP: MAKE SURE THIS IS NOT SET ON BY DEFAULT
                action_embeddings = self.embed_action(torch.zeros_like(actions))
            else:
                action_embeddings = self.embed_action(actions)
                action_embeddings = action_embeddings + time_embeddings

            # this makes the sequence look like (o1, s1, a1,o2, s2, a2, ...)
            # which works nice in an autoregressive sense since states predict actions
            # note that o1 and o2 need not be different
            stacked_inputs = torch.stack(
                (option_embeddings, state_embeddings, action_embeddings),
                dim=1).permute(
                0, 2, 1, 3).reshape(
                batch_size, 3 * seq_length, self.hidden_size)
            # LAYERNORM
            stacked_inputs = self.embed_ln(stacked_inputs)

            # to make the attention mask fit the stacked inputs, have to stack it as well
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

            # we feed in the input embeddings (not word indices as in NLP) to the model
            transformer_outputs = self.transformer(
                inputs_embeds=stacked_inputs,
                attention_mask=stacked_attention_mask,
            )
            x = transformer_outputs['last_hidden_state']

            # reshape x so that the second dimension corresponds to the original
            # options (0), states (1) or actions (2); i.e. x[:,0,t] is the token for s_t
            traj_out = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
            # get predictions
            # predict next state given option, state and action. skip the last state for prediction
            if isinstance(self.state_dim, int):
                state_preds = self.predict_state(traj_out[:, 2])[:, :-1, :]
            else:
                state_preds = None
            # predict next action given state and option
            action_preds = self.predict_action(traj_out[:, 1])

            # reconstruct current option given current option
            if self.option_il:
                option_preds = self.predict_option(traj_out[:, 0])
                options_loss = F.mse_loss(option_preds, options.detach())
            else:
                options_loss = None

            outputs = {'state_preds': state_preds,
                       'action_preds': action_preds,
                       'options_loss': options_loss}

            if self.predict_q:
                # predict next Q given state and option   ## IMP: Don't use current action
                q_preds = self.predict_q(traj_out[:, 1])
                outputs.update({'q_preds': q_preds})

            return outputs

        if self.use_language:
            assert word_embeddings is not None
            num_tokens = word_embeddings.shape[1]
            state_embeddings = self.embed_state(states)
            lang_embeddings = self.embed_lang(word_embeddings)
            action_embeddings = self.embed_action(actions)
            time_embeddings = self.embed_timestep(timesteps)

            # time embeddings are treated similar to positional embeddings
            state_embeddings = state_embeddings + time_embeddings
            action_embeddings = action_embeddings + time_embeddings

            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings),
                dim=1).permute(
                0, 2, 1, 3).reshape(
                batch_size, 2 * seq_length, self.hidden_size)
            lang_and_inputs = torch.cat([lang_embeddings, stacked_inputs], dim=1)
            # LAYERNORM AFTER LANGUAGE
            stacked_inputs = self.embed_ln(lang_and_inputs)

            # to make the attention mask fit the stacked inputs, have to stack it as well
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)
            lang_attn_mask = torch.cat(
                [torch.ones((batch_size, num_tokens), device=states.device), stacked_attention_mask], dim=1)

            # we feed in the input embeddings (not word indices as in NLP) to the model
            transformer_outputs = self.transformer(
                inputs_embeds=stacked_inputs,
                attention_mask=lang_attn_mask,
            )
            x = transformer_outputs['last_hidden_state']

            # reshape x so that the second dimension corresponds to the original
            # states (0), or actions (1); i.e. x[:,0,t] is the token for s_t
            lang_out = x[:, :num_tokens, :].reshape(
                batch_size, num_tokens, 1, self.hidden_size).permute(0, 2, 1, 3)
            traj_out = x[:, num_tokens:, :].reshape(
                batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

            # get predictions
            # predict state given state, action. skip the last prediction
            if isinstance(self.state_dim, int):
                state_preds = self.predict_state(traj_out[:, 1])[:, :-1, :]
            else:
                state_preds = None
            action_preds = self.predict_action(traj_out[:, 0])  # predict next action given state

            outputs = {'state_preds': state_preds,
                       'action_preds': action_preds}

            if self.predict_q:
                # predict next Q given state   ## IMP: Don't use current action
                q_preds = self.predict_q(traj_out[:, 0])
                outputs.update({'q_preds': q_preds})

            return outputs

    def get_action(self, states, actions, timesteps, options=None, word_embeddings=None, **kwargs):

        if self.use_options:
            assert options is not None
            if isinstance(self.state_dim, tuple):
                states = states.reshape(1, -1, *self.state_dim)
            else:
                states = states.reshape(1, -1, self.state_dim)
            options = options.reshape(1, -1, self.option_dim)
            actions = actions.reshape(1, -1, self.act_dim)
            timesteps = timesteps.reshape(1, -1)

            if self.max_length is not None:
                states = states[:, -self.max_length:]
                options = options[:, -self.max_length:]
                actions = actions[:, -self.max_length:]
                timesteps = timesteps[:, -self.max_length:]

                # pad all tokens to sequence length
                attention_mask = pad(torch.ones(1, states.shape[1]), self.max_length).to(
                    dtype=torch.long, device=states.device).reshape(1, -1)
                states = pad(states, self.max_length).to(dtype=torch.float32)
                options = pad(options, self.max_length).to(dtype=torch.float32)
                actions = pad(actions, self.max_length).to(dtype=torch.float32)
                timesteps = pad(timesteps, self.max_length).to(dtype=torch.long)
            else:
                raise ValueError('Should not have max_length NONE')
                attention_mask = None

            preds = self.forward(
                states, actions, timesteps, options=options, attention_mask=attention_mask)

        if self.use_language:
            assert word_embeddings is not None
            if isinstance(self.state_dim, tuple):
                states = states.reshape(1, -1, *self.state_dim)
            else:
                states = states.reshape(1, -1, self.state_dim)
            actions = actions.reshape(1, -1, self.act_dim)
            timesteps = timesteps.reshape(1, -1)

            if self.max_length is not None:
                states = states[:, -self.max_length:]
                actions = actions[:, -self.max_length:]
                timesteps = timesteps[:, -self.max_length:]

                # pad all tokens to sequence length
                attention_mask = pad(
                    torch.ones(1, states.shape[1]),
                    self.max_length).to(
                    dtype=torch.long, device=states.device).reshape(
                    1, -1)
                states = pad(states, self.max_length).to(dtype=torch.float32)
                actions = pad(actions, self.max_length).to(dtype=torch.float32)
                timesteps = pad(timesteps, self.max_length).to(dtype=torch.long)
            else:
                attention_mask = None

            preds = self.forward(
                states, actions, timesteps, word_embeddings=word_embeddings, attention_mask=attention_mask, **kwargs)

        return preds

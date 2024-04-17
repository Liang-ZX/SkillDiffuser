import numpy as np
import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from option_transformer import OptionTransformer
from utils import pad, entropy
from img_encoder import Encoder

from vector_quantize_pytorch import VectorQuantize


class OptionSelector(nn.Module):

    """
    This model takes in the language embedding and the state to output a z from a categorical distribution
    Use the VQ trick to pick an option z
    """

    def __init__(
            self, state_dim, num_options, option_dim, lang_dim, horizon, num_hidden=None, hidden_size=None,
            method='traj_option', option_transformer=None, codebook_dim=16, use_vq=True, kmeans_init=False,
            commitment_weight=0.25, **kwargs):

        # option_dim and codebook_dim are different because of the way the VQ package is designed
        # if they are different, then there is a projection operation that happens inside the VQ layer

        super().__init__()

        if num_hidden is not None:
            assert num_hidden >= 2, "We need at least two hidden layers!"

        self.state_dim = state_dim
        self.option_dim = option_dim
        self.use_vq = use_vq
        self.num_options = num_options

        self.horizon = horizon
        self.method = method   # whether to use full trajectory to get options or just current state
        self.hidden_size = 128
        hidden_size = 128

        if option_transformer:
            self.hidden_size = option_transformer.hidden_size

        self.Z = VectorQuantize(
            dim=option_dim,
            codebook_dim=codebook_dim,       # codebook vector size
            codebook_size=num_options,     # codebook size
            decay=0.99,             # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=commitment_weight,   # the weight on the commitment loss
            kmeans_init=kmeans_init,   # use kmeans init
            cpc=False,
            # threshold_ema_dead_code=2,  # should actively replace any codes that have an exponential moving average cluster size less than 2
            use_cosine_sim=False   # l2 normalize the codes
        )

        if self.method == 'traj_option':
            option_transformer_args = {'state_dim': state_dim,
                                       'lang_dim': lang_dim,
                                       'option_dim': option_dim,
                                       'hidden_size': option_transformer.hidden_size,
                                       'max_length': option_transformer.max_length,
                                       'max_ep_len': option_transformer.max_ep_len,
                                       'n_layer': option_transformer.n_layer,
                                       'n_head': option_transformer.n_head,
                                       'n_inner': 4*option_transformer.hidden_size,
                                       'activation_function': option_transformer.activation_function,
                                       'n_positions': option_transformer.n_positions,
                                       'resid_pdrop': option_transformer.dropout,
                                       'attn_pdrop': option_transformer.dropout,
                                       'output_attentions': True  # option_transformer.output_attention,
                                       }
            self.option_dt = OptionTransformer(**option_transformer_args)
        else:
            if isinstance(state_dim, tuple):
                # LORL
                if state_dim[0] == 3:
                    # LORL Sawyer
                    self.embed_state = Encoder(hidden_size=hidden_size, ch=3, robot=False)
                else:
                    # LORL Franka
                    self.embed_state = Encoder(hidden_size=hidden_size, ch=12, robot=True)
            else:
                self.embed_state = nn.Linear(state_dim, hidden_size)

            z_layers = []
            for i in range(num_hidden):
                if i == 0:
                    z_layers.append(nn.Linear(2*hidden_size, hidden_size))
                elif i == num_hidden-1:
                    z_layers.append(nn.Linear(hidden_size, option_dim))
                else:
                    z_layers.append(nn.Linear(hidden_size, hidden_size))
            self.pred_options = nn.Sequential(*z_layers)
            self.embed_lang = nn.Linear(lang_dim, hidden_size)

    def forward(self, word_embeddings, states, timesteps=None, attention_mask=None, **kwargs):
        if self.method == 'traj_option':
            dt_ret = self.option_dt(word_embeddings, states, timesteps, attention_mask)
            option_preds = dt_ret[0]
            state_embeddings = dt_ret[2]
            option_preds = option_preds[:, ::self.horizon, :]
        else:
            ret_state_embeddings = self.embed_state(states).clone().detach()
            horizon_states = states[:, ::self.horizon, :]
            state_embeddings = self.embed_state(horizon_states)
            lang_embeddings = self.embed_lang(word_embeddings)  # these will be cls embeddings or word embeddings mean

            inp = torch.cat([lang_embeddings.repeat(
                1, state_embeddings.shape[1], 1), state_embeddings], dim=-1)
            option_preds = self.pred_options(inp)

            state_embeddings = ret_state_embeddings

        if self.use_vq:
            options, indices, commitment_loss = self.Z(option_preds)
            entropies = entropy(self.Z.codebook, options, self.Z.project_in(option_preds))
        else:
            # TODO: For now simply return the first dim of option
            options, indices = option_preds, option_preds[:, :, 0]
            commitment_loss = None
            entropies = None
        return options, indices, commitment_loss, entropies, state_embeddings

    def get_option(self, word_embeddings, states, timesteps=None, **kwargs):

        if 'constant_option' in kwargs:
            return self.Z.project_out(
                self.Z.codebook[kwargs['constant_option']]), torch.tensor(
                kwargs['constant_option'])

        if self.method == 'traj_option':
            if isinstance(self.state_dim, tuple):
                states = states.reshape(1, -1, *self.state_dim)
            else:
                states = states.reshape(1, -1, self.state_dim)
            timesteps = timesteps.reshape(1, -1)
            max_length = self.option_dt.max_length

            if max_length is not None:
                states = states[:, -max_length:]
                timesteps = timesteps[:, -max_length:]

                # pad all tokens to sequence length
                attention_mask = pad(
                    torch.ones(1, states.shape[1]),
                    max_length).to(
                    dtype=torch.long, device=states.device).reshape(
                    1, -1)
                states = pad(states, max_length).to(dtype=torch.float32)
                timesteps = pad(timesteps, max_length).to(dtype=torch.long)
            else:
                attention_mask = None
                raise ValueError('Attention mask should not be none')

            options, option_indx, _, _, _ = self.forward(
                word_embeddings, states, timesteps, attention_mask=attention_mask, **kwargs)
        else:
            states = states[:, ::self.horizon, :]
            options, option_indx, _, _, _ = self.forward(
                word_embeddings, states, None, attention_mask=None, **kwargs)

        return options[0, -1], option_indx[0, -1]

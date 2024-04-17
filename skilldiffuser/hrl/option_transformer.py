import numpy as np
import torch
import torch.nn as nn
import transformers

from trajectory_gpt2 import GPT2Model
from img_encoder import Encoder


class OptionTransformer(nn.Module):

    """
    This model uses GPT-2 to select options for every horizon-th state
    """

    def __init__(
            self,
            state_dim,
            lang_dim,
            option_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            **kwargs):
        super().__init__()

        self.option_dim = option_dim
        self.hidden_size = hidden_size

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        self.state_dim = state_dim
        self.max_length = max_length
        self.output_attentions = kwargs["output_attentions"]

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

        self.embed_lang = nn.Linear(lang_dim, hidden_size)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_options = torch.nn.Linear(hidden_size, self.option_dim)

    def forward(self, word_embeddings, states, timesteps, attention_mask, **kwargs):
        batch_size, seq_length = states.shape[0], states.shape[1]
        num_tokens = word_embeddings.shape[1]

        if attention_mask is None:
            raise ValueError('Should not have attention_mask NONE')
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        ret_state_embeddings = state_embeddings.clone().detach()
        lang_embeddings = self.embed_lang(word_embeddings)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings  # (batch_size, seq_length, hidden)
        lang_and_inputs = torch.cat([lang_embeddings, state_embeddings], dim=1)
        # LAYERNORM AFTER LANGUAGE
        stacked_inputs = self.embed_ln(lang_and_inputs)

        lang_attn_mask = torch.cat([torch.ones((batch_size, num_tokens), device=states.device), attention_mask], dim=1)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=lang_attn_mask,
        )

        x = transformer_outputs['last_hidden_state']
        lang_out = x[:, :num_tokens, :].reshape(batch_size, num_tokens, self.hidden_size)
        traj_out = x[:, num_tokens:, :].reshape(batch_size, seq_length, self.hidden_size)

        # get predictions
        # predict option logits given state
        option_preds = self.predict_options(traj_out)

        if self.output_attentions:
            attentions = transformer_outputs[-1]
            return option_preds, attentions, ret_state_embeddings

        return option_preds, None, ret_state_embeddings

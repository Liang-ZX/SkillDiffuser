
from typing import Dict, Iterable, Callable
from collections import Counter
from itertools import chain
import numpy as np
import torch
import seaborn as sns
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
import wandb


class Attention(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._attention = None

        model.option_selector.option_dt.register_forward_hook(self.save_attention_hook())

    def save_attention_hook(self) -> Callable:
        def fn(model, input, output):
            self._attention = output[-1]
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._attention


def get_tokens(inputs, tokenizer):
    input_ids = inputs['input_ids']
    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)[1:]
    return tokens


def viz_matrix(words_dict, num_options, step, skip_words):
    # skip_words = ['go', 'to', 'the', 'a', '[SEP]']
    words = sorted(set(chain(*words_dict.values())) - set(skip_words))

    def w_to_ind(word):
        return words.index(word)

    matrix = np.zeros([len(words), num_options])

    for o in range(num_options):
        for w in words_dict[o]:
            if w not in skip_words:
                matrix[w_to_ind(w), o] += 1

    # plot co-occurence matrix (words x options)
    plt.figure(figsize=(30, 10))
    sns.heatmap(matrix, yticklabels=words)
    # plt.plot()
    wandb.log({"Correlation Matrix": wandb.Image(plt)}, step=step)

    # Now if we normalize it by column (word freq for each option)
    plt.figure(figsize=(30, 10))
    matrix_norm_col = (matrix)/(matrix.sum(axis=0, keepdims=True) + 1e-6)
    sns.heatmap(matrix_norm_col, yticklabels=words)
    wandb.log({"Word Freq Matrix": wandb.Image(plt)}, step=step)

    # Now if we normalize it by row (option freq for each word)
    plt.figure(figsize=(30, 10))
    matrix_norm_row = (matrix)/(matrix.sum(axis=1, keepdims=True) + 1e-6)
    sns.heatmap(matrix_norm_row, yticklabels=words)
    wandb.log({"Option Freq Matrix": wandb.Image(plt)}, step=step)
    plt.close()


def plot_hist(stats):
    plt.clf()
    plt.bar(stats.keys(), stats.values())
    plt.xticks(rotation='vertical')
    return wandb.Image(plt)

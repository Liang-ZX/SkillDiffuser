import torch
import torch.nn as nn
import torch.nn.functional as F


class StateReconstructor(nn.Module):
    """
    This model tries to reconstruct the states from which the options were chosen for each option
    """

    def __init__(self, option_dim, state_dim, num_hidden, hidden_size):
        super().__init__()

        assert num_hidden >= 2, "We need at least two hidden layers!"
        assert isinstance(state_dim, int), "State dimension has to be integer!"

        layers = []
        for i in range(num_hidden):
            if i == 0:
                layers.append(nn.Linear(option_dim, hidden_size))
            elif i == num_hidden-1:
                layers.append(nn.Linear(hidden_size, state_dim))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
        self.predictor = nn.Sequential(*layers)

    def forward(self, options):
        return self.predictor(options)


class LanguageReconstructor(nn.Module):
    """
    This model tries to reconstruct the language from all the options
    """

    def __init__(self, option_dim, max_options, lang_dim, num_hidden, hidden_size):
        super().__init__()

        assert num_hidden >= 2, "We need at least two hidden layers!"

        self.max_options = max_options

        layers = []
        for i in range(num_hidden):
            if i == 0:
                layers.append(nn.Linear(max_options*option_dim, hidden_size))
            elif i == num_hidden-1:
                layers.append(nn.Linear(hidden_size, lang_dim))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
        self.predictor = nn.Sequential(*layers)

    def forward(self, options):
        options = F.pad(options, pad=(1, self.max_options-options.shape[1]))
        return self.predictor(options)

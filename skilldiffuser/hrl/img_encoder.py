import torch
from torch import nn
from torch.nn import functional as F


# Image Encoder
# From https://github.com/suraj-nair-1/lorel/blob/main/models.py
class BaseEncoder(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self):
        super().__init__()


    def preprocess(self, observations):
        """
            Reshape to 4 dimensions so it works for convolutions
            Chunk the time and batch dimensions
        """
        B, T, C, H, W = observations.shape
        return observations.reshape(-1, C, H, W).type(torch.float32).contiguous()
        
    def unpreprocess(self, embeddings, B, T):
        """
            Reshape back to 5 dimensions 
            Unsqueeze the batch and time dimensions
        """
        BT, E = embeddings.shape
        return embeddings.reshape(B, T, E)
        

# Image Encoder
# From https://github.com/suraj-nair-1/lorel/blob/main/models.py
class Encoder(BaseEncoder):
    __constants__ = ['embedding_size']

    def __init__(self, hidden_size, activation_function='relu', ch=3, robot=False):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.robot = robot
        if self.robot:
            g = 4
        else:
            g = 1
        self.conv1 = nn.Conv2d(ch, 32, 4, stride=2, padding=1, groups=g)  # 3
        self.conv1_2 = nn.Conv2d(32, 32, 4, stride=1, padding=1, groups=g)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1, groups=g)
        self.conv2_2 = nn.Conv2d(64, 64, 4, stride=1, padding=1, groups=g)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1, groups=g)
        self.conv3_2 = nn.Conv2d(128, 128, 4, stride=1, padding=1, groups=g)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1, groups=g)
        self.conv4_2 = nn.Conv2d(256, 256, 4, stride=1, padding=1, groups=g)

        self.fc1 = nn.Linear(1024, 512)
        self.fc1_2 = nn.Linear(512, 512)
        self.fc1_3 = nn.Linear(512, 512)
        self.fc1_4 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, hidden_size)

    def forward(self, observations):
        if self.robot:
            observations = torch.cat([
                observations[:, :3], observations[:, 12:15], observations[:, 3:6], observations[:, 15:18],
                observations[:, 6:9], observations[:, 18:21], observations[:, 9:12], observations[:, 21:],
            ], 1)
        if len(observations.shape) == 5:
            preprocessed_observations = self.preprocess(observations)
        else:
            preprocessed_observations = observations
        hidden = self.act_fn(self.conv1(preprocessed_observations))
        hidden = self.act_fn(self.conv1_2(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv2_2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv3_2(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = self.act_fn(self.conv4_2(hidden))
        hidden = hidden.reshape(preprocessed_observations.shape[0], -1)

        hidden = self.act_fn(self.fc1(hidden))
        hidden = self.act_fn(self.fc1_2(hidden))
        hidden = self.act_fn(self.fc1_3(hidden))
        hidden = self.act_fn(self.fc1_4(hidden))
        hidden = self.fc2(hidden)
        
        if len(observations.shape) == 5:
            return self.unpreprocess(hidden, observations.shape[0], observations.shape[1])
        else:
            return hidden

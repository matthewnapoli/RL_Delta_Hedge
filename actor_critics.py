import torch
import torch.nn as nn
import torch.optim as optim

### ACTOR CLASS ###
class Actor(nn.Module):
    def __init__(self, stateDimension, hidden=128):
        """
            Feedforward actor network mapping states to normalized actions

            Two hidden layers with ReLU activations followed by a Tanh output layer produce a scalar in [−1, 1]
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(stateDimension, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, tensorOfStates):
        return self.net(tensorOfStates)

## CRITIC CLASSES ##
class Critic(nn.Module):
    def __init__(self, stateDimension, hidden=128):
        """
            Q‑function network mapping state–action pairs to scalars
            Takes a concatenated tensor [s, a] and outputs Q(s,a)
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(stateDimension + 1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)

class SecondMomentCritic(nn.Module):
    """
        Critic for second moment: approximates Q2(s,a) = E[G^2 | s,a]
    """
    def __init__(self, stateDimension, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(stateDimension + 1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)
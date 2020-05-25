import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, learning_rate=1e-3):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        interm1 = 100
        interm2 = 50
        # define a simple feed forward network
        self.model = nn.Sequential(
            nn.Linear(state_size, interm1),
            nn.ReLU(),
            nn.Linear(interm1,interm2),
            nn.ReLU(),
            nn.Linear(interm2,action_size)
            )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.MSELoss(reduction='sum')

    def forward(self, state):
        """Build a network that maps state -> action values."""
        action_pred = self.model(state)
        return action_pred


import numpy as np
import torch

DEVICE = torch.device("cpu")

class ReplayBuffer:
    """
        Circular replay buffer for experience tuples.

        Stores transitions (s, a, r, s', done) and allows random sampling of
        minibatches for stochastic gradient descent.
    """
    def __init__(self, capacity, stateDimension, actionDimension=1):
        """
            Preallocates NumPy arrays: current/next states, actions, rewards, "done" flags
            Tracks: where the next write goes (pointer), how full the buffer currently is (size)

            capacity: max number of transitions to store (eg. 200,000)
            stateDimension: dimension of preprocessed state (here 4)
            actionDimension: dimension of action (here 1 because hedge position is scalar)
        """
        self.capacity = int(capacity)
        self.stateDimension = stateDimension
        self.actionDimension = actionDimension
        self.pointer = 0
        self.size = 0

        self.currentStates = np.zeros((capacity, stateDimension), dtype=np.float32)
        self.actions = np.zeros((capacity, actionDimension), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.nextStates = np.zeros((capacity, stateDimension), dtype=np.float32)
        self.doneFlags = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, currentStates, actions, rewards, nextStates, doneFlags):
        """
            Store a transition at the current pointer and advance the pointer
        """
        self.currentStates[self.pointer] = currentStates
        self.actions[self.pointer] = actions
        self.rewards[self.pointer] = rewards
        self.nextStates[self.pointer] = nextStates
        self.doneFlags[self.pointer] = float(doneFlags)

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batchSize):
        """
            Randomly selects indices among stored transitions and returns a batch as torch tensors on the correct device.

            batchSize: number of transitions to sample
        """
        idx = np.random.randint(0, self.size, size=batchSize)
        batch = {"currentStates": torch.tensor(self.currentStates[idx], device=DEVICE),
                 "actions": torch.tensor(self.actions[idx], device=DEVICE),
                 "rewards": torch.tensor(self.rewards[idx], device=DEVICE),
                 "nextStates": torch.tensor(self.nextStates[idx], device=DEVICE),
                 "doneFlags": torch.tensor(self.doneFlags[idx], device=DEVICE)
                }
        return batch
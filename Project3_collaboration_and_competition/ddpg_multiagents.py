import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg_agent import Agent

BUFFER_SIZE = int(1e6)  ### replay buffer size
BATCH_SIZE = 256        ### minibatch size
GAMMA = 0.99            ### discount factor
LEARNING_TIMES = 3      ### the number of times for learning at a single phase.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgents:
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed=10):
        super(MultiAgents, self).__init__()
        """Initialize an MultiAgent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.ddpg_agents = [Agent(state_size, action_size), Agent(state_size, action_size)]
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def act(self, observations):
        """get actions from all agents in the MADDPG object"""

        actions = [agent.act(obs) for agent, obs in zip(self.ddpg_agents,observations)]
        return actions

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        ### Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
    def trigger_learn(self):
        ### Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            ### for each agent, learn LEARNING_TIMES
            for agent in self.ddpg_agents:
                for i in range(LEARNING_TIMES):
                    experiences = self.memory.sample()
                    agent.learn(experiences)

                    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  ### internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
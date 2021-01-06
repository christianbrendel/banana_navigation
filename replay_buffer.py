import random
from collections import namedtuple, deque
import numpy as np

class ReplayBuffer:
    
    def __init__(self, buffer_size, e=None, a=None):
        """Initialize the replay buffer. Define e and a for prioritized replay."""
        self.buffer_size = buffer_size
        self.a = a
        self.e = e
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "td_error"])
        self.memory = deque(maxlen=self.buffer_size)
    
    def add(self, state, action, reward, next_state, done, td_error=np.nan):
        """Add an experience tuple to the replay buffer."""
        exp = self.experience(state, action, reward, next_state, done, td_error)
        self.memory.append(exp)
    
    def sample(self, k):
        """Sample k experiences from the replat buffer."""
        if self.a is None or self.e is None:
            experiences = random.sample(self.memory, k=k)
            sampling_probs = 1/len(self.memory) * np.ones(k)
        else:
            raise NotImplementedError('Prioritized Replay is not yet implemented!')
            
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])
        sampling_probs = np.vstack([s for s in sampling_probs if s is not None])
        
        return (states, actions, rewards, next_states, dones, sampling_probs)
    
    def __len__(self):
        return len(self.memory)
"""
PPO Buffer模块
"""

from typing import List


class PPOBuffer:
    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self.clear()
    
    def add(self, state: List[int], action: int, reward: float, 
            log_prob: float, value: float, done: bool = False):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def get_batch(self):
        return {
            'states': self.states.copy(),
            'actions': self.actions.copy(),
            'rewards': self.rewards.copy(),
            'log_probs': self.log_probs.copy(),
            'values': self.values.copy(),
            'dones': self.dones.copy()
        }
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def __len__(self):
        return len(self.states)
    
    def is_full(self):
        return len(self.states) >= self.max_size

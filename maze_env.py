import numpy as  np
from .my_env_lib import get_env_info
from .environment import GridWorldEnv


class MazeEnv(GridWorldEnv):
    def __init__(self, grid, move_prov=1.0, rewards=None):
        if grid==None:
            grids, experts = get_env_info()
            grid = grids[0] # Learn reward of first agent
            self._expert = experts[0][0]
        super().__init__(grid, move_prov)
        self._states = super().states
        self.n_states = len(self._states)
        self.n_actions = len(super().actions)
        self.trans_probs = self._get_trans_probs()
        self._rewards = rewards if rewards else np.zeros(self.n_states) 
        self.state = super().start_pos 

    @property
    def rewards(self):
        return self._rewards

    @property
    def expert(self):
        return self._expert

    @rewards.setter
    def rewards(self, rewards):
        if isinstance(rewards, list):
            assert len(rewards) == self.n_states, 'Invalid rewards specified'
            rewards = np.array(rewards)
        assert rewards.shape == (self.n_states,), 'Invalid rewards specified'
        self._rewards = rewards

    def step(self, a):
        self.state = super()._move(self.state, a)
        reward = self._get_reward(self.state)
        return self.state, reward

    def _get_reward(self, state=None):
        return self.rewards[state]

    def _get_trans_probs(self):
        return self.trans_probs

    def reset(self):
        self.state = super().start_pos
    
    # trans_probs(move prov=1.0)
    def _get_trans_probs(self):
        trans_probs = np.zeros((1, self.n_states*self.n_actions*self.n_states))
        trans_probs = trans_probs.reshape(self.n_states, self.n_actions, self.n_states)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                ns = self._move(s, a)
                trans_probs[s][a][ns] = 1.0
        return trans_probs



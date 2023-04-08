import numpy as np
from gym.envs.toy_text import discrete
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
from pprint import pprint
from colorama import Fore, Back, Style

class GridWorldEnv(discrete.DiscreteEnv):

    def __init__(self, grid, move_prob=1.0):
        # grid is 2d-array, and each value treated as attribute.
        # attribute is
        #  S: reward cell (game start)
        #  0: ordinary cell
        #  -1: damage cell (game end)
        #  G: reward cell (game end)
        self.grid = grid
        if isinstance(grid, (list, tuple)):
            self.grid = np.array(grid)

        self._actions = {
            "UP": 0,
            "LEFT": 1,
            "DOWN": 2,
            "RIGHT": 3,
        }
        
        self.move_prob = move_prob
        num_states = self.nrow * self.ncol
        num_actions = len(self._actions)

        # Setting start position
        initial_state_prob = np.zeros(num_states)
        initial_state_prob[self.start_pos] = 1.0
        self.reward_func = np.zeros(num_states)
        # Make transitions
        P = {}
        for s in range(num_states):
            if s not in P:
                P[s] = {a : [] for a in range(num_actions)}

            reward = self.reward_func[s]
            done = self.has_done(s)
            if done:
                # Terminal state
                for a in range(num_actions):
                    P[s][a] +=[(1.0,s,reward)]
            else:
                for a in range(num_actions):
                    transition_probs = self.transit_func(s, a)
                    for n_s in transition_probs:
                        if transition_probs[n_s] == 0.0:
                            continue
                        reward = self.reward_func[n_s]
                        done = self.has_done(s)
                        P[s][a] +=[(transition_probs[n_s],n_s,reward)]
        self.P = P
        super().__init__(num_states, num_actions, P, initial_state_prob)

    @property
    def nrow(self):
        return self.grid.shape[0]

    @property
    def ncol(self):
        return self.grid.shape[1]

    @property
    def shape(self):
        return self.grid.shape

    @property
    def actions(self):
        return list(range(self.action_space.n))

    @property
    def states(self):
        return list(range(self.observation_space.n))
    
    @property
    def start_pos(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                if 'S' == self.grid[row][col]:
                    index = self.coordinate_to_state(row, col)
                    break;
        return index

    @property
    def goal_pos(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                if 'G' == self.grid[row][col]:
                    index = self.coordinate_to_state(row, col)
                    break;
        return index


    def state_to_coordinate(self, s):
        row, col = divmod(s, self.ncol)
        return row, col

    def coordinate_to_state(self, row, col):
        index = row * self.ncol + col
        return index

    def state_to_feature(self, s):
        feature = np.zeros(self.observation_space.n)
        feature[s] = 1.0
        return feature

    
    def transit_func(self, state, action):
        transition_probs = {}
        #opposite_direction = (action + 2) % 4
        candidates = [a for a in range(len(self._actions))]
    
        for a in candidates:
            prob = 0
            if a == action:
                prob = self.move_prob
            else:
                prob = (1 - self.move_prob) / 2
            if prob==0:
                continue

            row, col = self.state_to_coordinate(state)
            next_row, next_col = row, col
            # Move state by action
            if a == self._actions["LEFT"]:
                next_col -= 1
            elif a == self._actions["DOWN"]:
                next_row += 1
            elif a == self._actions["RIGHT"]:
                next_col += 1
            elif a == self._actions["UP"]:
                next_row -= 1

            is_out_range = False
            # Check the out of grid
            if not (0 <= next_row < self.nrow):
                is_out_range = True
            if not (0 <= next_col < self.ncol):           
                is_out_range = True
            if (not is_out_range) and self.grid[next_row][next_col] == '-1':
                is_out_range = True
            


            if not is_out_range:
                next_state = self.coordinate_to_state(next_row, next_col)
                #next_state = self._move(state, a)
                if next_state not in transition_probs:
                    transition_probs[next_state] = prob
                else:
                    transition_probs[next_state] += prob
        return transition_probs

    def action_to_vec(self, action):
        action_vec = [0, 0]
        if action == self._actions["UP"]:
            action_vec = [0, -1]
        elif action == self._actions["LEFT"]:
            action_vec = [-1, 0]
        elif action == self._actions["DOWN"]:
            action_vec = [0, 1]
        elif action == self._actions["RIGHT"]:
            action_vec = [1, 0]
        return action_vec

    # 即時報酬の付与
    def get_reward(self, state):
        reward = self.reward_func[state]
        return reward

    # goalについたか？
    def has_done(self, state):
        row, col = self.state_to_coordinate(state)
        goal = self.grid[row][col]
        if goal == 'G':
            return True
        else:
            return False

    def _move(self, state, action):
        next_state = state
        row, col = self.state_to_coordinate(state)
        next_row, next_col = row, col
        
        # Move state by action
        if action == self._actions["LEFT"]:
            next_col -= 1
        elif action == self._actions["DOWN"]:
            next_row += 1
        elif action == self._actions["RIGHT"]:
            next_col += 1
        elif action == self._actions["UP"]:
            next_row -= 1

        # Check the out of grid
        if not (0 <= next_row < self.nrow):
            next_row, next_col = row, col
        if not (0 <= next_col < self.ncol):           
            next_row, next_col = row, col
        if self.grid[next_row][next_col] == '-1':
            next_row, next_col = row, col
        
        return self.coordinate_to_state(next_row, next_col)

    def move_no_wall(self, state, action):
        next_state = state
        row, col = self.state_to_coordinate(state)
        next_row, next_col = row, col
        
        # Move state by action
        if action == self._actions["LEFT"]:
            next_col -= 1
        elif action == self._actions["DOWN"]:
            next_row += 1
        elif action == self._actions["RIGHT"]:
            next_col += 1
        elif action == self._actions["UP"]:
            next_row -= 1

        if not (0 <= next_row < self.nrow):
            next_row, next_col = row, col
        if not (0 <= next_col < self.ncol):           
            next_row, next_col = row, col
            
        return self.coordinate_to_state(next_row, next_col)
        
        return next_state
    def is_wall(self, s):
        row, col = self.state_to_coordinate(s)
        if self.grid[row][col] == '-1':
            return True
        else:
            return False

    def is_wall_traj(self, traj):
        for s in traj:
            if self.is_wall(s):
                return True
        return False

    def print_env(self):
        for row in self.grid:
            for s in row:
                if s=='S':
                    print(Fore.BLUE+s, end='')
                    print(Style.RESET_ALL, end='')
                elif s=='G':
                    print(Fore.RED+s, end='')
                    print(Style.RESET_ALL, end='')                    
                elif s=='-1':
                    print(Fore.YELLOW+"!", end='')
                    print(Style.RESET_ALL, end='')   
                else:
                    print(s, end='')
                print(" ", end='')
            print("\n", end='')
    
    def plot_on_grid(self, values, folder = "Non"):
        if len(values.shape) < 2:
            values = values.reshape(self.shape)
        plt.figure()
        sns.heatmap(values,annot= True,square=True,cmap='PuRd')
        if folder != "Non":
            save_dir = folder+ str(datetime.datetime.now().strftime('%Y-%m-%d'))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            fileName = str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f')) +'.png'
            plt.savefig(os.path.join(save_dir, fileName))
            plt.show() 

    def create_expert(self):
        s = self.start_pos
        g = self.goal_pos
        s_row, s_col = self.state_to_coordinate(s)
        g_row, g_col = self.state_to_coordinate(g)
        expert = [s]
        e = [s_row, s_col]

        count = 0
        x = 1 if s_row<g_row else -1
        while e[0] != g_row:
            e[0] += x
            e_state = self.coordinate_to_state(e[0], e[1])
            expert += [e_state]
            if count>self.nrow * self.ncol:
                print("out of range in create_expert")
                break  

        count = 0
        x = 1 if s_col<g_col else -1
        while e[1]!=g_col:
            e[1] += x
            e_state = self.coordinate_to_state(e[0], e[1])
            expert += [e_state]
            if count>self.nrow * self.ncol:
                print("out of range in create_expert")
                break  

        return expert

    def create_expert_trajectories(self, state_trajs):
        trajectories = []
        for trajs in state_trajs:
            trajectory = []
            for i_s in range(1, len(trajs)):
                n_sx, n_sy = self.state_to_coordinate(trajs[i_s])
                sx,sy = self.state_to_coordinate(trajs[i_s-1])
                #self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
                if sx+1 == n_sx and sy == n_sy:
                    action_int = 0                       
                elif sx == n_sx and sy+1 == n_sy:
                    action_int = 1
                elif sx-1 == n_sx and sy == n_sy:
                    action_int = 2
                elif sx == n_sx and sy-1 == n_sy:
                    action_int = 3
                else:
                    action_int = 0
                    
                state_int = trajs[i_s-1]
                trajectory.append((state_int, action_int, 0))
            trajectory.append((trajs[-1], 0, 0))
            trajectories.append(trajectory)
        return np.array(trajectories)
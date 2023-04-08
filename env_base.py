import numpy as np
import random
import copy
import configparser
import csv,difflib,copy
import pandas as pd
import json
import configparser
from pprint import pprint
from .environment import GridWorldEnv

def state_to_coordinate(s, col):
    row, col = divmod(s, col)
    return row, col

def coordinate_to_state(nrow, row, col):
    index = row * nrow + col
    return index

def get_random_states(num, state):
    random_states = random.sample(range(state[0]*state[1]), num)
    
    random_coordinate = []
    for s in random_states:
        row, col = state_to_coordinate(s, state[1])
        random_coordinate += [[row, col]]
    return random_coordinate

def create_env(state_size, agents, obstacles=None):
    env = [[] for _ in range(len(agents))]
    print(len(agents))
    print(agents)
    for i in range(len(agents)):
        print(i)
        e = [[0]*state_size[1] for _ in range(state_size[0])]
        e[agents[i][0][0]][agents[i][0][1]] = 'S'
        e[agents[i][1][0]][agents[i][1][1]] = 'G'
        if obstacles:
            for o in obstacles:
                if o:
                    e[o[0]][o[1]] = '-1'
        env[i] = GridWorldEnv(grid=e)
    return env

def create_random_env(N_AGENTS, STATE_SIZE, N_OBSTACLES=0):
    start_state = []
    goal_state = []
    obstacles = []
    
    start_state = get_random_states(N_AGENTS, STATE_SIZE)
    while True:
        goal_state = get_random_states(N_AGENTS, STATE_SIZE)
        if all([x!=y for x,y in zip(start_state, goal_state)]):
            break
    # ここ後で直す（重複する可能性）
    tf = True
    while tf:
        obstacles = get_random_states(N_OBSTACLES, STATE_SIZE)
        tf = False
        for o in obstacles:
            if any([s==o for s in start_state]):
                tf = True
            if any([g==o for g in goal_state]):
                tf = True
                    
    print(start_state)
    print(goal_state)

    env = [[] for _ in range(N_AGENTS)]
    for i in range(N_AGENTS):
        e = [[0]*STATE_SIZE[1] for i in range(STATE_SIZE[0])]
        e[start_state[i][0]][start_state[i][1]] = 'S'
        e[goal_state[i][0]][goal_state[i][1]] = 'G'
        if obstacles:
            for o in obstacles:
                if o:
                    e[o[0]][o[1]] = '-1'
        env[i] = GridWorldEnv(grid=e)
    return env


def create_cycle_env(state_size):
    state_size = np.array(state_size)
    if state_size[0]!=state_size[1]:
        min_state = np.min(state_size)
        print(f'Change state_size ({state_size[0]}, {state_size[1]}) -> ({min_state}, {min_state})')
        state_size = np.array([min_state, min_state])

    def right_cycle(width, pos):
        if pos[0]==width[0]:
            add = np.array([0, 1]) if pos[1]!=width[1] else np.array([1, 0])
        elif pos[0]==width[1]:
            add = np.array([0, -1]) if pos[1]!=width[0] else np.array([-1, 0])
        elif pos[1]==width[0]:
            add = np.array([-1, 0]) if pos[0]!=width[0] else np.array([0, 1])
        elif pos[1]==width[1]:
            add = np.array([1, 0]) if pos[0]!=width[1] else np.array([0, -1])
        else:
            print('Cannot classify (in right_cycle).')
        return np.array(pos)+add

    def left_cycle(width, pos):
        if pos[0]==width[0]:
            add = np.array([0, -1]) if pos[1]!=width[0] else np.array([1, 0])
        elif pos[0]==width[1]:
            add = np.array([0, 1]) if pos[1]!=width[1] else np.array([-1, 0])
        elif pos[1]==width[0]:
            add = np.array([1, 0]) if pos[0]!=width[1] else np.array([0, 1])
        elif pos[1]==width[1]:
            add = np.array([-1, 0]) if pos[0]!=width[0] else np.array([0, -1])
        else:
            print('Cannot classify (in left_cycle).')
        return np.array(pos)+add

    n_rc = state_size[0]
    agents = [[] for _ in range(n_rc**2)] if n_rc%2==0 else [[] for _ in range(n_rc**2)]
    for i in range(n_rc):
        for j in range(n_rc):
            if n_rc%2==1 and i==int(n_rc/2) and j==int(n_rc/2):
                continue
            agents[i*n_rc+j].append([i, j]) # start pos
            agents[i*n_rc+j].append([n_rc-i-1, n_rc-j-1]) # goal pos
            
            pos = np.array([i, j])
            goal = np.array([n_rc-i-1, n_rc-j-1])
            width = np.min(np.array([i, j, n_rc-i-1, n_rc-j-1]))
            width = [width, n_rc-width-1]
            expert = [coordinate_to_state(n_rc, pos[0], pos[1])]
            if j<n_rc/2:
                while any(pos!=goal):
                    pos = right_cycle(width, pos)
                    expert.append(coordinate_to_state(n_rc, pos[0], pos[1]))
            else:
                while any(pos!=goal):
                    pos = left_cycle(width, pos)
                    expert.append(coordinate_to_state(n_rc, pos[0], pos[1]))
            agents[i*n_rc+j].append(expert)
    return agents

def create_my_env(state_size):
    state_size = np.array(state_size)
    if state_size[0]!=state_size[1]:
        min_state = np.min(state_size)
        print(f'Change state_size ({state_size[0]}, {state_size[1]}) -> ({min_state}, {min_state})')
        state_size = np.array([min_state, min_state])

    def right_cycle(width, pos):
        if pos[0]==width[0]:
            add = np.array([0, 1]) if pos[1]!=width[1] else np.array([1, 0])
        elif pos[0]==width[1]:
            add = np.array([0, -1]) if pos[1]!=width[0] else np.array([-1, 0])
        elif pos[1]==width[0]:
            add = np.array([-1, 0]) if pos[0]!=width[0] else np.array([0, 1])
        elif pos[1]==width[1]:
            add = np.array([1, 0]) if pos[0]!=width[1] else np.array([0, -1])
        else:
            print('Cannot classify (in right_cycle).')
        return np.array(pos)+add

    def left_cycle(width, pos):
        if pos[0]==width[0]:
            add = np.array([0, -1]) if pos[1]!=width[0] else np.array([1, 0])
        elif pos[0]==width[1]:
            add = np.array([0, 1]) if pos[1]!=width[1] else np.array([-1, 0])
        elif pos[1]==width[0]:
            add = np.array([1, 0]) if pos[0]!=width[1] else np.array([0, 1])
        elif pos[1]==width[1]:
            add = np.array([-1, 0]) if pos[0]!=width[0] else np.array([0, -1])
        else:
            print('Cannot classify (in left_cycle).')
        return np.array(pos)+add

    n_rc = state_size[0]
    agents = []
    for i in range(1, n_rc-1):
        for j in range(1, n_rc-1):
            if n_rc%2==1 and i==int(n_rc/2) and j==int(n_rc/2):
                continue
            agent = []
            agent.append([i, j]) # start pos
            agent.append([n_rc-i-1, n_rc-j-1]) # goal pos
            
            pos = np.array([i, j])
            goal = np.array([n_rc-i-1, n_rc-j-1])
            width = np.min(np.array([i, j, n_rc-i-1, n_rc-j-1]))
            width = [width, n_rc-width-1]
            expert = [coordinate_to_state(n_rc, pos[0], pos[1])]
            if j<n_rc/2:
                while any(pos!=goal):
                    pos = right_cycle(width, pos)
                    expert.append(coordinate_to_state(n_rc, pos[0], pos[1]))
            else:
                while any(pos!=goal):
                    pos = left_cycle(width, pos)
                    expert.append(coordinate_to_state(n_rc, pos[0], pos[1]))
            agent.append(expert)
            agents.append(copy.deepcopy(agent))

    pos = np.array([0, 0])
    expert = [coordinate_to_state(state_size[0], pos[0], pos[1])]
    act_right = True 
    while any(pos!=state_size-1):
        add = np.array([0, 1]) if act_right else np.array([1, 0]) 
        pos += add 
        expert.append(coordinate_to_state(state_size[0], pos[0], pos[1]))
        act_right = False if act_right else True
    agents.append([[0, 0], [state_size[0]-1,state_size[1]-1], expert])
    return agents



def get_env_info():
    ACTION = "ACTION"
    MAIRL_PARAM = "MAIRL_PARAM"

    config_ini = configparser.ConfigParser()
    config_ini.optionxform = str
    config_ini.read('../../config/config.ini', encoding='utf-8')
    N_ACTIONS = int(config_ini.get(ACTION, "N_ACTIONS"))
    action_set = json.loads(config_ini.get(ACTION, "ACTION_SET"))
    ENV = json.loads(config_ini.get("ENV", "ENV_INFO")) 

    experts = []
    start_goal_position = []
    N_AGENTS = int(config_ini.get(ENV, "N_AGENTS"))
    STATE_SIZE = json.loads(config_ini.get(ENV, "STATE_SIZE")) 
    obstacle = json.loads(config_ini.get(ENV, "OBSTACLE")) 
    for i in range(N_AGENTS):
        agent_info = json.loads(config_ini.get(ENV,"AGENT_START_GOAL_EXPERT"+str(i+1)))
        start_goal_position += [agent_info[0]]
        experts += [agent_info[1]]

    grids = [[] for i in range(N_AGENTS)]
    for i in range(N_AGENTS):
        grid = [[0]*STATE_SIZE[1] for i in range(STATE_SIZE[0])]
        grid[start_goal_position[i][0][0]][start_goal_position[i][0][1]] = 'S'
        grid[start_goal_position[i][1][0]][start_goal_position[i][1][1]] = 'G'
        for o in obstacle:
            if o:
                grid[o[0]][o[1]] = '-1'
        grids[i] = grid
    return grids, experts 

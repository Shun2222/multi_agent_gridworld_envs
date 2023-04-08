import configparser
import csv,difflib,copy
import pandas as pd
import json
import configparser
import pickle
from .environment import GridWorldEnv

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

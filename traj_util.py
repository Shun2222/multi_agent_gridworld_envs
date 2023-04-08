import numpy as np

#重複がなければFlase
def has_duplicates(seq):
    return len(seq) != len(set(seq)) # setが重複を許さないためseqに重複があればsetは小さくなる
    
def exist_traj(traj, n):
    t = np.ones(n)*traj[-1]
    for i in range(len(traj)):
        t[i] = traj[i]
    return t

def is_collision(traj1, traj2):
    if not traj2 or not traj1:
        return False
    if len(traj1)!=len(traj2):
        if len(traj1)<len(traj2):
            traj1 = exist_traj(traj1, len(traj2))
        else:
            traj2 = exist_traj(traj2, len(traj1))
    min_len = min([len(traj1), len(traj2)])
    np_traj1 = np.array(traj1)
    np_traj2 = np.array(traj2)
    if any(np_traj1[:min_len] == np_traj2[:min_len]):
        return True
    elif any([True if x and y else False for x,y in zip((np_traj1[1:min_len] == np_traj2[0:min_len-1]),(np_traj1[0:min_len-1] == np_traj2[1:min_len]))]):
        return True
    return False

def is_collision_trajs(trajs):
    is_col = [False]*len(trajs)
    for i in range(len(trajs)):
        for j in range(i+1, len(trajs)):
            if is_collision(trajs[i], trajs[j]):
                is_col[i] = True
                is_col[j] = True
    return is_col

def is_collision_matrix(trajs):
    is_col_mat = [[False]*len(trajs) for _ in range(len(trajs))]
    for i in range(len(trajs)):
        for j in range(i+1, len(trajs)):
            if is_collision(trajs[i], trajs[j]):
                is_col_mat[i][j] = True
                is_col_mat[j][i] = True
    return is_col_mat

def  array_to_str(arr):
    return ','.join(map(str, arr))

def str_to_array(str_arr):
    return [int(s.strip()) for s in str_arr.split(',')]

def traj_to_action_vecs(traj, state_size):
    action_vecs = np.zeros([state_size[0]*state_size[1],2])
    for i in range(len(traj)-1):
        s = np.array(divmod(traj[i], state_size[1]))
        ns = np.array(divmod(traj[i+1], state_size[1]))
        action_vecs[traj[i]] = ns-s
    return action_vecs

def calc_state_visition_count(n_state, trajs):
    features = np.zeros(n_state)
    for t in trajs:
        for s in t:
            features[s] += 1
    return features

def states_to_feature(n_state, traj):
    return calc_state_visition_count(n_state, [traj])

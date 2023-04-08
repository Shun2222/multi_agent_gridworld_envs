import pickle
import os 
import datetime
import numpy as np

def make_path(save_dir="./", file_name="Non", extension=".png"):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if file_name == "Non":
        file_name = str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f')) 
    file_name += extension
    return os.path.join(save_dir, file_name)

def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

def mean_pre_nex(log_file, key, num_pre=5, num_nex=5):
    log = pickle_load(log_file)
    data = log[key]
    m_data = np.mean(data, axis=0)[1:]
    ave_data = []
    for i in np.arange(len(m_data)):
        pre = i-num_pre if i-num_pre>=0 else 0
        nex = i+num_nex if i+num_nex<len(m_data) else len(m_data)-1
        ave_data += [np.mean(m_data[pre:nex])]
    return ave_data  

class Logger():
    def __init__(self):
        self.save_dir=None
        self.datas=None

    def set_save_dir(self, save_dir):
        make_path(save_dir)
        self.save_dir = save_dir

    def set_datas(self, datas):
        self.datas = datas

    def dump(self, file_name):
        with open(os.path.join(self.save_dir, file_name), mode='wb') as f:
            pickle.dump(self.datas, f)
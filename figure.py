import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches
import seaborn as sns
import datetime
import os
import numpy as np
from MAIRL.environment import *
from MAIRL.libs.traj_util import *
from MAIRL.libs.data_handle import *



def plot_on_grid(values, state_size, file_name="Non", folder="./", set_annot=True, save=True, show=False, title=""):
    values = np.array(values)
    if len(values.shape) < 2:
        values = values.reshape(state_size)
    plt.figure()
    plt.title(title)
    img = sns.heatmap(values,annot=set_annot,square=True,cmap='PuRd')
    if save:
        file_path = make_path(folder, file_name)
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.close()
    return img

def ax_heatmap(values, state_size, ax, decimals=2, set_annot=True, cbar=True, square=True, label_size=6, title=""):
    values = np.array(values)
    if len(values.shape) < 2:
        values = values.reshape(state_size)
    values = np.round(values, decimals=decimals)
    sns.heatmap(values, ax=ax, annot=set_annot,cbar=cbar,square=square,cmap='PuRd')  
    ax.tick_params(axis='x', labelsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)
    ax.set_title(title)

def arrow_plot(data, actions, state_size, file_name="Non", folder="./", save=True, show=False, title=""):
    data = np.array(data)
    if len(data.shape) < 2:
        data = data.reshape(state_size)
    actions = np.array(actions)
    if len(actions.shape) < 2:
        actions = actions.reshape(state_size)
    e=[['0']*state_size[1] for _ in range(state_size[0])]
    e[0][0] = 'S'
    e[state_size[0]-1][state_size[1]-1]='G'
    env = GridWorldEnv(grid=e)
    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (12, 8))
    sns.heatmap(data, ax=ax, cbar=True, cbar_ax=cbar_ax, annot=True, square=True)
    ax.set_title(title)
    for s in range(state_size[0]*state_size[1]):
         y, x = divmod(s, state_size[1])
         if actions[y][x]==-1:
            continue
         a = np.array(env.action_to_vec(actions[y][x]))
         ax.arrow(x+0.5, state_size[0]-1.5+y, a[0]*0.6, a[1]*0.6, color='blue', head_width=0.2, head_length=0.2)
    file_path = make_path(folder, file_name)
    if save:
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.close()

def trajs_plot(trajs, state_size, file_name="Non", folder="./", save=True, show=False, title=""):
    action_vecs = [[] for _ in range(len(trajs))]
    for i in range(len(trajs)):
        action_vecs[i]  = traj_to_action_vecs(trajs[i], state_size)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = [8,8])
    ax.set_title(title)
    ax.grid(True)
    ax.set_xticks(np.arange(state_size[1]+1))
    ax.set_yticks(np.arange(state_size[0]+1))
    ax.set_xlim(0, state_size[1])
    ax.set_ylim(0, state_size[0])
    cmap = plt.cm.jet
    cNorm  = colors.Normalize(vmin=0, vmax=len(trajs))
    scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)
    for i in range(len(trajs)):
        colorVal = scalarMap.to_rgba(i)
        y, x = divmod(trajs[i][0], state_size[1])
        c = patches.Circle( xy=(x+0.5+0.35, state_size[0]-1-y+0.5+0.35), radius=0.1, color=colorVal)
        ax.text(x+0.5+0.25, state_size[0]-1-y+0.5+0.25, str(i), size=8)
        ax.add_patch(c)
        for s in range(state_size[0]*state_size[1]):
            if all(action_vecs[i][s]==[0,0]):
                continue
            a = action_vecs[i][s]
            y, x = divmod(s, state_size[1])
            ax.arrow(x+0.5, state_size[0]-1-y+0.5, a[1]*0.3, -a[0]*0.3, color=colorVal, head_width=0.2, head_length=0.2)
    file_path = make_path(folder, file_name)
    if save:
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.close()

class make_gif():
    def __init__(self):
        self.datas = []

    def add_data(self, d):
        self.datas += [d]

    def add_datas(self, ds):
        self.datas += ds

    def make(self, state_size, folder="./", file_name="Non", save=True, show=False):
        def make_heatmap(i):
            ax.cla()
            ax.set_title("Iteration="+str(i))
            data = np.array(self.datas[i])
            if len(data.shape) < 2:
                data = data.reshape(state_size)
            sns.heatmap(data, ax=ax, cbar=True, cbar_ax=cbar_ax)
        fms = len(self.datas) if len(self.datas)<=128 else np.linspace(0, len(self.datas)-1, 128).astype(int)
        grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
        fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (12, 8))
        ani = animation.FuncAnimation(fig=fig, func=make_heatmap, frames=fms, interval=500, blit=False)
        if save:
            file_path = make_path(folder, file_name, extension=".gif")
            ani.save(file_path, writer="pillow")
        if show:
            plt.show() 
        plt.close()

    def reset(self):
        plt.close()
        self.datas = []

    def make_test(self, state_size, folder="./", file_name="Non", save=True, show=False):
        def make_arrow(i, state_size):
            ax.cla()
            ax.set_title("Iteration="+str(i))
            trajs = self.datas[i]
            for i in range(len(trajs)):
                action_vecs[i]  = traj_to_action_vecs(trajs[i], state_size)
            ax.grid(True)
            ax.set_xticks(np.arange(state_size[1]+1))
            ax.set_yticks(np.arange(state_size[0]+1))
            ax.set_xlim(0, state_size[1])
            ax.set_ylim(0, state_size[0])
            for i in range(len(trajs)):
                colorVal = scalarMap.to_rgba(i)
                for s in range(state_size[0]*state_size[1]):
                    if all(action_vecs[i][s]==[0,0]):
                        continue
                    a = action_vecs[i][s]
                    y, x = divmod(s, state_size[1])
                    ax.arrow(x+0.5, state_size[0]-1-y+0.5, a[1]*0.3, -a[0]*0.3, color=colorVal, head_width=0.2, head_length=0.2)


        fms = len(self.datas) if len(self.datas)<=128 else np.linspace(0, len(self.datas)-1, 128).astype(int)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = [8,8])
        cmap = plt.cm.jet
        cNorm  = colors.Normalize(vmin=0, vmax=fms)
        scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)
        ani = animation.FuncAnimation(fig=fig, func=make_arrow, fargs=(state_size), frames=fms, interval=500, blit=False)
        if save:
            file_path = make_path(folder, file_name, extension=".gif")
            ani.save(file_path, writer="pillow")
        if show:
            plt.show() 
        plt.close()        
    
def plot_steps_seeds(save_dirs, label=""):
    plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("step")
    data = []
    for d in save_dirs:
        ave_step = mean_pre_nex(log_file=d+"/logs.pickle", key="step_in_multi_hist")
        data.append(ave_step)
    data = np.array(data)
    m = data.mean(axis=0)
    std = data.std(axis=0) 

    plt.fill_between(np.arange(len(m)), m+std, m-std, alpha=0.2)
    plt.plot(np.arange(len(m)), m, label=label)

    fileName = "step_in_multi_mean_seeds"+'.png'
    if label!="":
        plt.legend()
    plt.savefig(save_dirs[0]+"/"+fileName) 
    plt.close()


def heatmap_gif(datas, state_size, feature=False, labels=None, folder="./", file_name="Non", save=True, show=False):
    def make_heatmap(i):
        print(f"{i+1}/{fms}")
        for j in range(len(datas[-1])):
            ax = fig.add_subplot(fig_size[0], fig_size[1], j+1)
            ax.cla()
            if labels:
                ax.set_title(labels[i])
            else:
                ax.set_title(str(i))
            data = np.array(datas[i][j])
            if not feature:
                data = calc_state_visition_count(n_state, [data])
            if len(data.shape) < 2:
                data = data.reshape(state_size)
            sns.heatmap(data, ax=ax, cbar=False, square=True, annot=False)
            ax.axis('off')

    print(f"{len(datas[-1])} <= n*m")
    s = input("n,m = ").split(",")
    fig_size = [int(s[0]), int(s[1])]
    n_state = state_size[0]*state_size[1]
    fms = len(datas) if len(datas)<=128 else np.linspace(0, len(datas)-1, 128).astype(int)
    
    plt.axis('off')
    fig= plt.figure()
    plt.subplots_adjust(wspace=0.4, hspace=0.4, right=0.9, left=0.1, top=0.9, bottom=0.1)
    plt.axis('off')
    ani = animation.FuncAnimation(fig=fig, func=make_heatmap, frames=fms, interval=500, blit=False)
    if save:
        file_path = make_path(folder, file_name, extension=".gif")
        ani.save(file_path, writer="pillow")
    if show:
        plt.show() 
    plt.close()
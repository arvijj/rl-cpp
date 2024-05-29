import csv
import numpy as np
import os
import random
import torch
from matplotlib import pyplot as plt


def seed_everything(seed, env=None):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.action_space.np_random.seed(seed)
        env.observation_space.np_random.seed(seed)

def get_logs(log_file, as_dict=False):
    steps, rewards, ep_lengths, times, levels = [], [], [], [], []
    with open(log_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i < 2: # skip first two rows (headers)
                continue
            rewards.append(float(row[0]))
            ep_lengths.append(int(row[1]))
            times.append(float(row[2]))
            levels.append(float(row[3]))
            steps.append((steps[-1] if len(steps) != 0 else 0) + ep_lengths[-1])
    if as_dict:
        return {'steps': steps,
                'reward': rewards,
                'ep_length': ep_lengths,
                'time': times,
                'level': levels}
    return steps, rewards, ep_lengths, times, levels

def smooth(values, smoothing):
    assert isinstance(values, list)
    k = smoothing
    padded = [values[0]]*k + values + [values[-1]]*k
    return np.convolve(padded, [1/(k*2 + 1)]*(k*2 + 1), mode='valid').tolist()

def plot_results(exp_path, x_axis='steps', smoothing=None):
    _plot(exp_path, x_axis, smoothing)
    plt.show()

def save_plot_as_pdf(exp_path, x_axis='steps', smoothing=None):
    _plot(exp_path, x_axis, smoothing)
    plt.savefig(os.path.join(exp_path, 'plot_train.pdf'), format='pdf', bbox_inches='tight')

def _plot(exp_path, x_axis, smoothing):
    assert x_axis in ['steps', 'time']
    logs = get_logs(os.path.join(exp_path, 'logs.monitor.csv'), as_dict=True)

    # Smooth the y values
    x_vals = logs[x_axis]
    reward = logs['reward']
    ep_length = logs['ep_length']
    level = logs['level']
    if smoothing is not None:
        reward = smooth(reward, smoothing)
        ep_length = smooth(ep_length, smoothing)

    # Plot
    fig, ax1 = plt.subplots()
    ax1.plot(x_vals, reward, label='reward')
    ax1.plot(x_vals, ep_length, label='ep length')
    ax1.legend(loc='upper left')
    ax1.set_ylabel('reward & episode length')
    ax1.set_xlabel(x_axis)
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(x_vals, level, '-k', label='level')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('level')
    fig.tight_layout()

def total_variation(img, img2=None, mode='sym-iso'):
    """
    Computes the total variation of img.
    If img2 is specified, the pixel variation is computed in both img and
    max(img, img2), where the minimum value of these two are kept in each pixel.
    Modes:
      sym-iso: Symmetric and isotropic (slowest)
      non-sym-iso: Isotropic but not symmetric (faster than sym-iso)
      non-iso: Non-isotropic and symmetric (fastest)
    TODO: support any shapes, i.e. (n,) and (n,m), where n,m >= 1
    """
    assert mode in ['sym-iso', 'non-sym-iso', 'non-iso']
    img = img.astype(float)
    diff1 = np.abs(img[1:, :] - img[:-1, :])
    diff2 = np.abs(img[:, 1:] - img[:, :-1])
    if img2 is not None:
        assert img.shape == img2.shape
        img2 = np.maximum(img, img2)
        diff1 = np.minimum(diff1, np.abs(img2[1:, :] - img2[:-1, :]))
        diff2 = np.minimum(diff2, np.abs(img2[:, 1:] - img2[:, :-1]))
    if mode == 'sym-iso':
        tv = np.sum(np.sqrt(diff1[:, 1:] ** 2 +  diff2[1:, :] ** 2)) + \
             np.sum(np.sqrt(diff1[:, 1:] ** 2 +  diff2[:-1, :] ** 2)) + \
             np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[:-1, :] ** 2)) + \
             np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[1:, :] ** 2))
        return tv / 4
    elif mode == 'non-sym-iso':
        return np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[:-1, :] ** 2))
    else:
        return np.sum(diff1) + np.sum(diff2)

def format_float_str(number, decimals, spaces):
    """
    Formats a float to a string to fit in a limited number of characters.
    Example: 3.14159265 --> " 3.14" with decimals=2 and spaces=5
    """
    assert decimals >= 0
    assert spaces >= 0
    if spaces == 0:
        return ""
    if np.isnan(number):
        s = str(number).rjust(spaces)
    elif decimals == 0:
        s = str(int(round(float(number), 0)))
        s = s.rjust(spaces)
    else:
        spaces_int = max(1, spaces - decimals - 1)
        s = str(round(float(number), decimals))
        s1, s2 = s.split('.')
        s = s1.rjust(spaces_int) + '.' + s2.ljust(decimals)
    return s[:spaces]

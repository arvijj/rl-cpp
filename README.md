# RL Coverage Path Planning

Code implementation of the conference paper [__Learning Coverage Paths in Unknown Environments with Deep Reinforcement Learning, ICML, 2024__](https://proceedings.mlr.press/v235/jonnarth24a.html), and its journal extension [__Sim-to-Real Transfer of Deep Reinforcement Learning Agents for Online Coverage Path Planning, IEEE Access, 2025__](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11039829).

Lawn mowing | Exploration
:-------------------------:|:-------------------------:
<img src="figures/mowing_path.png" alt="mowing path" height="300" /> | <img src="figures/exploration_path.png" alt="exploration path" height="300" />

WITHOUT total variation reward | WITH total variation reward
:-------------------------:|:-------------------------:
<img src="figures/mowing_no_tv.png" alt="without total variation reward" width="300" /> | <img src="figures/mowing_tv.png" alt="with total variation reward" width="300" />

## Install

* [Install miniconda](https://docs.conda.io/en/latest/miniconda.html)
* `conda create -n rlm python=3.9`
* `conda activate rlm`
* `pip install setuptools==65.5.0 pip==21`
* `pip install wheel==0.38.0`
* (CPU) `pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`
* (GPU) `pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116`

## Run

Train an RL agent for CPP in a simulated 2D environment (reduce `--buffer_size` if memory is limited):
* Mowing: `python train.py --logdir my-experiment`
* Exploration: `python train.py --logdir my-experiment --exploration --local_tv_reward_scale 0.2 --no-overlap_observation --no-steering_limits_lin_vel`

Check how the trained agent performs:
* Plot logs: `python plot.py --load my-experiment`
* Render: `python eval.py --load my-experiment`

Run full quantitative evaluation and show the resulting paths and metrics:
* Run eval: `python eval.py --load my-experiment --verbose --no-render --metrics_dir metrics`
* Show path: `python show_path.py --metrics_dir my-experiment/metrics --type eval --episode 1`

## Pre-trained weights

Pre-trained weights can be found at [this link](https://drive.google.com/file/d/1hJXxRQTaFd0MEuB2UuQT9D6I-p_QXiAM/view?usp=sharing). The zip file contains three models:
* __exploration__: An exploration CPP model configured for Explore-Bench.
* __mowing_tv1__: A CPP model for the lawn mowing task, with an incremental TV reward of 1.
* __mowing_tv2__: Same as _mowing_tv1_, but with TV reward = 2. Makes slightly nicer looking patterns, but has slower coverage times on average.

Each folder contains a metrics folder. Visualize the paths like this:
* `python show_path.py --metrics_dir weights/mowing_tv1/metrics --type eval --episode 1`

## Tests

Run tests:
* Run `pytest` from the root folder.

## Cite

Conference paper:
```
@InProceedings{jonnarth2024icml,
  title = {Learning Coverage Paths in Unknown Environments with Deep Reinforcement Learning},
  author = {Jonnarth, Arvi and Zhao, Jie and Felsberg, Michael},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series = {Proceedings of Machine Learning Research},
  year = {2024},
  volume = {235},
  pages = {22491--22508},
}
```

Journal extension:
```
@article{jonnarth2025access,
  title = {Sim-to-Real Transfer of Deep Reinforcement Learning Agents for Online Coverage Path Planning},
  author = {Jonnarth, Arvi and Johansson, Ola and Zhao, Jie and Felsberg, Michael},
  journal = {IEEE Access},
  year = {2025},
  volume = {13},
  pages = {106883-106905},
}
```

import argparse
import importlib
import json
import os
from argparse import BooleanOptionalAction
from datetime import datetime
from stable_baselines3.common.monitor import Monitor

from rlm.architectures import StackedMapFeaturesExtractor
from rlm.mower_env import MowerEnv


def main():
    parser = argparse.ArgumentParser()
    agent_args = parser.add_argument_group('agent')
    agent_args.add_argument('--algo', default='SAC', type=str)
    agent_args.add_argument('--learning_rate', default=2e-5, type=float)
    agent_args.add_argument('--cnn', default=True, action=BooleanOptionalAction)
    agent_args.add_argument('--cnn_dims', default=256, type=int)
    agent_args.add_argument('--grouped_convs', default=True, action=BooleanOptionalAction)
    agent_args.add_argument('--buffer_size', default=500_000, type=int)
    agent_args.add_argument('--train_freq', default=1, type=int)
    agent_args.add_argument('--gradient_steps', default=1, type=int)
    train_args = parser.add_argument_group('train')
    train_args.add_argument('--checkpoint', default=None, type=str)
    train_args.add_argument('--steps', default=1_000_000, type=int)
    train_args.add_argument('--logdir', default=None, type=str)
    env_args = parser.add_argument_group('env')
    env_args.add_argument('--input_size', default=32, type=int)
    env_args.add_argument('--num_maps', default=4, type=int)
    env_args.add_argument('--scale_factor', default=4, type=float)
    env_args.add_argument('--meters_per_pixel', default=0.0375, type=float)
    env_args.add_argument('--min_size', default=None, type=int)
    env_args.add_argument('--max_size', default=None, type=int)
    env_args.add_argument('--stacks', default=1, type=int)
    env_args.add_argument('--step_size', default=0.5, type=float)
    env_args.add_argument('--constant_lin_vel', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--max_lin_vel', default=0.26, type=float)
    env_args.add_argument('--max_ang_vel', default=1.0, type=float)
    env_args.add_argument('--max_lin_acc', default=None, type=float)
    env_args.add_argument('--max_ang_acc', default=None, type=float)
    env_args.add_argument('--action_delay', default=0, type=float)
    env_args.add_argument('--steering_limits_lin_vel', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--mower_radius', default=0.15, type=float)
    env_args.add_argument('--lidar_rays', default=24, type=int)
    env_args.add_argument('--lidar_range', default=3.5, type=float)
    env_args.add_argument('--lidar_fov', default=345, type=float)
    env_args.add_argument('--position_noise', default=0.01, type=float)
    env_args.add_argument('--heading_noise', default=0.05, type=float)
    env_args.add_argument('--lidar_noise', default=0.05, type=float)
    env_args.add_argument('--exploration', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--overlap_observation', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--frontier_observation', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--action_observations', default=0, type=int)
    env_args.add_argument('--eval', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--p_use_known_obstacles', default=0.7, type=float)
    env_args.add_argument('--p_use_unknown_obstacles', default=0.7, type=float)
    env_args.add_argument('--p_use_floor_plans', default=0.7, type=float)
    env_args.add_argument('--max_known_obstacles', default=100, type=int)
    env_args.add_argument('--max_unknown_obstacles', default=100, type=int)
    env_args.add_argument('--obstacle_radius', default=0.25, type=float)
    env_args.add_argument('--all_unknown', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--max_episode_steps', default=None, type=int)
    env_args.add_argument('--max_non_new_steps', default=1000, type=int)
    env_args.add_argument('--collision_ends_episode', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--flip_when_stuck', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--max_stuck_steps', default=5, type=int)
    env_args.add_argument('--start_level', default=1, type=int)
    env_args.add_argument('--use_goal_time_in_levels', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--goal_coverage', default=0.9, type=float)
    env_args.add_argument('--goal_coverage_reward', default=0, type=float)
    env_args.add_argument('--wall_collision_reward', default=-10, type=float)
    env_args.add_argument('--obstacle_collision_reward', default=-10, type=float)
    env_args.add_argument('--newly_visited_reward_scale', default=1, type=float)
    env_args.add_argument('--newly_visited_reward_max', default=2, type=float)
    env_args.add_argument('--overlap_reward_scale', default=0, type=float)
    env_args.add_argument('--overlap_reward_max', default=5, type=float)
    env_args.add_argument('--overlap_reward_always', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--local_tv_reward_scale', default=1, type=float)
    env_args.add_argument('--local_tv_reward_max', default=5, type=float)
    env_args.add_argument('--global_tv_reward_scale', default=0, type=float)
    env_args.add_argument('--global_tv_reward_max', default=5, type=float)
    env_args.add_argument('--use_known_obstacles_in_tv', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--use_unknown_obstacles_in_tv', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--frontier_reward_scale', default=0, type=float)
    env_args.add_argument('--frontier_reward_max', default=5, type=float)
    env_args.add_argument('--turn_reward_scale', default=0, type=float)
    env_args.add_argument('--obstacle_dilation', default=9, type=int)
    env_args.add_argument('--constant_reward', default=-0.1, type=float)
    env_args.add_argument('--constant_reward_always', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--truncation_reward_scale', default=0, type=float)
    env_args.add_argument('--coverage_pad_value', default=0, type=int)
    env_args.add_argument('--obstacle_pad_value', default=1, type=int)
    env_args.add_argument('--verbose', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--metrics_dir', default=None, type=str)
    args = parser.parse_args()
    assert args.algo in ['SAC', 'PPO'], 'Only SAC/PPO algorithms currently supported'
    print(args, flush=True)

    # Create dict of argument groups
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    # Create log directory and save parameters
    if args.logdir is not None:
        logdir = args.logdir
    else:
        logdir = os.path.join('experiments', datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    if arg_groups['env'].metrics_dir is not None:
        arg_groups['env'].metrics_dir = os.path.join(logdir, arg_groups['env'].metrics_dir)
    os.makedirs(logdir)
    with open(os.path.join(logdir, 'agent_parameters.json'), 'w') as f:
        json.dump(arg_groups['agent'].__dict__, f, indent=2)
    with open(os.path.join(logdir, 'train_parameters.json'), 'w') as f:
        json.dump(arg_groups['train'].__dict__, f, indent=2)
    with open(os.path.join(logdir, 'env_parameters.json'), 'w') as f:
        json.dump(arg_groups['env'].__dict__, f, indent=2)

    # Construct policy kwargs in case of CNN architecture
    if args.cnn:
        if 'SAC' in args.algo:
            net_arch = dict(
                pi=[args.cnn_dims, args.cnn_dims],
                qf=[args.cnn_dims, args.cnn_dims])
        elif args.algo == 'PPO':
            net_arch = [dict(
                pi=[args.cnn_dims, args.cnn_dims],
                vf=[args.cnn_dims, args.cnn_dims])]
        policy_kwargs = dict(
            net_arch=net_arch,
            features_extractor_class=StackedMapFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=args.cnn_dims,
                map_size=args.input_size,
                num_maps=args.num_maps,
                lidar_rays=args.lidar_rays,
                stacks=args.stacks,
                grouped_convs=args.grouped_convs,
                frontier_observation=args.frontier_observation))
    else:
        policy_kwargs = None

    # Train agent
    env = MowerEnv(**vars(arg_groups['env']))
    env = Monitor(env, os.path.join(logdir, 'logs'), info_keywords=('level',))
    algo = getattr(importlib.import_module('stable_baselines3'), args.algo)
    if args.checkpoint is not None:
        # TODO: also load parameters.json from previous run
        model = algo.load(args.checkpoint, env=env)
    else:
        kwargs = dict(verbose=1, policy_kwargs=policy_kwargs)
        if args.buffer_size is not None and 'SAC' in args.algo:
            kwargs['buffer_size'] = args.buffer_size
        if args.learning_rate is not None:
            kwargs['learning_rate'] = args.learning_rate
        if 'SAC' in args.algo:
            kwargs['train_freq'] = args.train_freq
            kwargs['gradient_steps'] = args.gradient_steps
        model = algo("MultiInputPolicy", env, **kwargs)
    model.learn(total_timesteps=args.steps)
    model.save(os.path.join(logdir, 'agent'))
    env.close()

if __name__ == '__main__':
    main()

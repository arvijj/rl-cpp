import argparse
import glob
import imageio
import importlib
import json
import math
import numpy as np
import os
import time
from argparse import BooleanOptionalAction
from matplotlib import pyplot as plt

import rlm.utils
from rlm.mower_env import MowerEnv
from rlm.random_mower import RandomMower
from rlm.spiral_mower import SpiralMower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--random', default=False, action=BooleanOptionalAction)
    parser.add_argument('--spiral', default=False, action=BooleanOptionalAction)
    parser.add_argument('--exploration', default=False, action=BooleanOptionalAction)
    parser.add_argument('--steps', default=50000, type=int)
    parser.add_argument('--max_non_new_steps', default=5000, type=int)
    parser.add_argument('--goal_coverage', default=[0.9, 0.99], type=float, nargs='+')
    parser.add_argument('--collision_ends_episode', default=False, action=BooleanOptionalAction)
    parser.add_argument('--flip_when_stuck', default=True, action=BooleanOptionalAction)
    parser.add_argument('--plot', default=False, action=BooleanOptionalAction)
    parser.add_argument('--save_pdf', default=False, action=BooleanOptionalAction)
    parser.add_argument('--save_time_series', default=False, action=BooleanOptionalAction)
    parser.add_argument('--buffer_size', default=1000, type=int)
    parser.add_argument('--render', default=True, action=BooleanOptionalAction)
    parser.add_argument('--render_mode', default='limited', type=str)
    parser.add_argument('--save_video', default=None, type=str)
    parser.add_argument('--video_speedup', default=1, type=float)
    parser.add_argument('--save_path_interval', default=None, type=int)
    parser.add_argument('--verbose', default=False, action=BooleanOptionalAction)
    parser.add_argument('--metrics_dir', default=None, type=str)
    args = parser.parse_args()
    assert args.load is not None or args.random or args.spiral, \
        'Either --load, --random, or --spiral needs to be specified'
    if not args.render:
        assert args.save_video is None, 'Need to specify --render to save video'
        assert args.save_path_interval is None, 'Need to specify --render to save path'
    assert args.save_video is None or args.save_path_interval is None, 'Saving both video and path is not supported'

    # Load parameters
    if args.load is not None:
        with open(os.path.join(args.load, 'agent_parameters.json')) as f:
            agent_args = argparse.Namespace(**json.load(f))
        with open(os.path.join(args.load, 'env_parameters.json')) as f:
            env_args = argparse.Namespace(**json.load(f))
            env_args.max_episode_steps = args.steps
            env_args.max_non_new_steps = args.max_non_new_steps
            env_args.eval = True
            env_args.goal_coverage = max(args.goal_coverage)
            env_args.collision_ends_episode = args.collision_ends_episode
            env_args.flip_when_stuck = args.flip_when_stuck
            env_args.verbose = args.verbose
            if args.metrics_dir is None:
                env_args.metrics_dir = None
            else:
                env_args.metrics_dir = os.path.join(args.load, args.metrics_dir)
            args.exploration = env_args.exploration # override exploration argument

    # Get number of eval maps
    if args.exploration:
        eval_maps = glob.glob('maps/eval_exploration*')
    else:
        eval_maps = glob.glob('maps/eval_mowing*')
    episodes = len(eval_maps)

    # Print evaluation settings
    print('Evaluation settings:', flush=True)
    print('  episodes:              ', episodes, flush=True)
    print('  steps per episode:     ', args.steps, flush=True)
    print('  goal coverage:         ', args.goal_coverage, flush=True)
    print('  collision ends episode:', args.collision_ends_episode, flush=True)
    print('  flip when stuck:       ', args.flip_when_stuck, flush=True)

    # Create env and agent
    if args.load is not None:
        env = MowerEnv(**vars(env_args))
    else:
        env = MowerEnv(
            num_maps=2,
            max_episode_steps=args.steps,
            max_non_new_steps=args.max_non_new_steps,
            exploration=args.exploration,
            eval=True,
            goal_coverage=max(args.goal_coverage),
            collision_ends_episode=args.collision_ends_episode,
            flip_when_stuck=args.flip_when_stuck,
            constant_lin_vel=False,
            steering_limits_lin_vel=False,
            verbose=args.verbose,
            metrics_dir=args.metrics_dir)
    if args.random:
        model = RandomMower(env)
    elif args.spiral:
        model = SpiralMower(env)
    else:
        algo = getattr(importlib.import_module('stable_baselines3'), agent_args.algo)
        model = algo.load(os.path.join(args.load, 'agent'), env=env, buffer_size=args.buffer_size)

    # Keep track of metrics
    steps = np.zeros((episodes, args.steps))
    times = np.zeros((episodes, args.steps))
    lengths = np.zeros((episodes, args.steps))
    turns = np.zeros((episodes, args.steps))
    coverages = np.zeros((episodes, args.steps))
    overlaps = np.zeros((episodes, args.steps))
    collisions = np.zeros((episodes, args.steps))

    # Create video writer
    if args.render and args.save_video is not None:
        writer = imageio.get_writer(
            args.save_video,
            fps=args.video_speedup/env.step_size)

    # Loop over episodes
    t_steps = 0
    t_tot_model = 0
    t_tot_env = 0
    data = dict()
    for ep in range(episodes):
        obs = env.reset()
        path_length = 0
        rotations = 0
        position_m_old = env.position_m.copy()
        heading_old = env.heading % (2 * math.pi)

        # Loop over steps
        for step in range(args.steps):

            # Compute model/env inference times
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            t_tot_model += time.time() - t0
            t0 = time.time()
            obs, reward, done, info = env.step(action)
            t_tot_env += time.time() - t0
            t_steps += 1

            # Render
            if args.render:
                if args.save_video:
                    img = env.render(mode='rgb_array')
                    writer.append_data(img)
                elif args.save_path_interval:
                    env.render(mode='limited')
                    if step % args.save_path_interval == 0 or done:
                        plt.savefig(f'path_{ep}ep_{step}step.pdf')
                else:
                    env.render(mode=args.render_mode)

            # Update path length
            path_length += np.linalg.norm(env.position_m - position_m_old)
            position_m_old = env.position_m.copy()

            # Update accumulated rotations
            heading_new = env.heading % (2 * math.pi)
            heading_diff = abs(heading_new - heading_old)
            heading_diff = min(heading_diff, 2 * math.pi - heading_diff)
            assert heading_diff >= 0 and heading_diff <= 2 * math.pi
            rotations += heading_diff / (2 * math.pi)
            heading_old = heading_new

            # Gather metrics
            steps[ep, step] = env.elapsed_steps
            times[ep, step] = env.elapsed_time
            lengths[ep, step] = path_length
            turns[ep, step] = rotations
            coverages[ep, step] = env.coverage_in_percent
            overlaps[ep, step] = env.overlap_in_percent
            collisions[ep, step] = env.num_collisions
            if done:
                steps[ep, step:] = steps[ep, step]
                times[ep, step:] = times[ep, step]
                lengths[ep, step:] = lengths[ep, step]
                turns[ep, step:] = turns[ep, step]
                coverages[ep, step:] = coverages[ep, step]
                overlaps[ep, step:] = overlaps[ep, step]
                collisions[ep, step:] = collisions[ep, step]
                break

        # Save the data
        if args.save_time_series and args.load is not None:
            data[env.filename] = dict()
            data[env.filename]['steps'] = steps[ep]
            data[env.filename]['time'] = times[ep]
            data[env.filename]['length'] = lengths[ep]
            data[env.filename]['turns'] = turns[ep]
            data[env.filename]['coverage'] = coverages[ep]
            data[env.filename]['overlap'] = overlaps[ep]
            data[env.filename]['collisions'] = collisions[ep]
    env.close()

    # Close video writer
    if args.render and args.save_video is not None:
        writer.close()

    # Compute metrics for fixed number of steps
    avg_model_infer_time = round(1000 * t_tot_model / t_steps, 2)
    avg_env_infer_time = round(1000 * t_tot_env / t_steps, 2)
    avg_coverage = round(100 * np.mean(coverages[:, -1]), 2)
    std_coverage = round(100 * np.std(coverages[:, -1], ddof=1), 2)

    # Compute metrics for fixed goal coverages
    coverage_reached_rate = []
    avg_steps, avg_time, avg_length, avg_turns, avg_overlap = [], [], [], [], []
    avg_collisions, collision_rate = [], []
    for gc in args.goal_coverage:
        eps = coverages[:, -1] >= gc # episodes where goal coverage was reached
        idxs = np.argmax(coverages[eps] >= gc, axis=1) # indices (steps) where coverage was reached
        coverage_reached_rate.append(round(100 * eps.mean(), 2))
        avg_steps.append(round(steps[eps][range(len(idxs)), idxs].mean(), 2))
        avg_time.append(round(times[eps][range(len(idxs)), idxs].mean(), 2))
        avg_length.append(round(lengths[eps][range(len(idxs)), idxs].mean(), 2))
        avg_turns.append(round(turns[eps][range(len(idxs)), idxs].mean(), 2))
        avg_overlap.append(round(100 * overlaps[eps][range(len(idxs)), idxs].mean(), 2))
        avg_collisions.append(round(collisions[eps][range(len(idxs)), idxs].mean(), 2))
        collision_rate.append(round(100 * (collisions[eps][range(len(idxs)), idxs] > 0).mean(), 2))

    # Print metrics
    print(f'Average metrics for {args.steps} steps:')
    print('  model infer time:      ', avg_model_infer_time, 'ms')
    print('  env infer time:        ', avg_env_infer_time, 'ms')
    print('  coverage:              ', avg_coverage, '+/-', std_coverage, '%')
    print(f'Average metrics at goal coverages:')
    print('  goal coverage:         ', args.goal_coverage)
    print('  coverage reached rate: ', coverage_reached_rate, '%')
    print('  steps:                 ', avg_steps, 'steps')
    print('  time:                  ', avg_time, 'seconds')
    print('  path length:           ', avg_length, 'meters')
    print('  accumulated turn:      ', avg_turns, 'full rotations')
    print('  overlap:               ', avg_overlap, '%')
    print('  number of collisions:  ', avg_collisions, 'collisions')
    print('  collision rate:        ', collision_rate, '%')

    # Print compact
    label = '|cover|'
    score = f'|{rlm.utils.format_float_str(avg_coverage, 2, 5)}|'
    for gc, gcrr in zip(args.goal_coverage, coverage_reached_rate):
        lbl = str(int(100*gc))
        label += 'G' + lbl + ' '*max(0, 3-len(lbl)) + '|'
        score += f'{rlm.utils.format_float_str(gcrr, 1, 4)}|'
    for gc, gcrr, t in zip(args.goal_coverage, coverage_reached_rate, avg_time):
        lbl = str(int(100*gc))
        label += 'T' + lbl + ' '*max(0, 3-len(lbl)) + '|'
        if gcrr == 100:
            score += f'{rlm.utils.format_float_str(t, 1, 4)}|'
        else:
            score += '  - |'
    if args.load is not None:
        logs = rlm.utils.get_logs(os.path.join(args.load, 'logs.monitor.csv'), as_dict=True)
        label += 'lvl|'
        score += f'{rlm.utils.format_float_str(logs["level"][-1], 0, 3)}|'
    print(label)
    print(score)

    # Save time series data
    if args.save_time_series and args.load is not None:
        np.save(os.path.join(args.load, 'eval_time_series.npy'), data)

    # Plot data
    if args.plot or (args.save_pdf and args.load is not None):
        fig, ax = plt.subplots(2, 3, figsize=(10, 6), layout='constrained')
        mean_time = times.mean(axis=0)
        mean_coverage = 100 * coverages.mean(axis=0)
        mean_length = lengths.mean(axis=0)
        mean_turns = turns.mean(axis=0)
        mean_overlap = 100 * overlaps.mean(axis=0)
        mean_collisions = collisions.mean(axis=0)
        std_coverage = 100 * coverages.std(axis=0, ddof=1)
        std_length = lengths.std(axis=0, ddof=1)
        std_turns = turns.std(axis=0, ddof=1)
        std_overlap = 100 * overlaps.std(axis=0, ddof=1)
        std_collisions = collisions.std(axis=0, ddof=1)
        ax[0, 0].plot(mean_coverage, mean_length)
        ax[0, 0].set_xlabel('coverage [%]')
        ax[0, 0].set_ylabel('path length [m]')
        ax[0, 1].plot(mean_time, mean_coverage)
        ax[0, 1].fill_between(mean_time, mean_coverage - std_coverage, mean_coverage + std_coverage, color='b', alpha=.1)
        ax[0, 1].set_xlabel('time [s]')
        ax[0, 1].set_ylabel('coverage [%]')
        ax[0, 2].plot(mean_time, mean_length)
        ax[0, 2].fill_between(mean_time, mean_length - std_length, mean_length + std_length, color='b', alpha=.1)
        ax[0, 2].set_xlabel('time [s]')
        ax[0, 2].set_ylabel('path length [m]')
        ax[1, 0].plot(mean_time, mean_turns)
        ax[1, 0].fill_between(mean_time, mean_turns - std_turns, mean_turns + std_turns, color='b', alpha=.1)
        ax[1, 0].set_xlabel('time [s]')
        ax[1, 0].set_ylabel('full rotations')
        ax[1, 1].plot(mean_time, mean_overlap)
        ax[1, 1].fill_between(mean_time, mean_overlap - std_overlap, mean_overlap + std_overlap, color='b', alpha=.1)
        ax[1, 1].set_xlabel('time [s]')
        ax[1, 1].set_ylabel('overlap [%]')
        ax[1, 2].plot(mean_time, mean_collisions)
        ax[1, 2].fill_between(mean_time, mean_collisions - std_collisions, mean_collisions + std_collisions, color='b', alpha=.1)
        ax[1, 2].set_xlabel('time [s]')
        ax[1, 2].set_ylabel('collisions')

        # Plot on screen
        if args.plot:
            plt.show()

        # Save as pdf
        if args.save_pdf and args.load is not None:
            plt.savefig(os.path.join(args.load, 'plot_eval.pdf'), format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()

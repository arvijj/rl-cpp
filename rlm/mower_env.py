import cv2
import glob
import gym
import math
import numpy as np
import os
import random
import time
from gym import spaces
from matplotlib import pyplot as plt
from matplotlib import transforms as mtransforms
from PIL import Image, ImageDraw, ImageFont

from rlm import utils

class MowerEnv(gym.Env):
    """
    An environment where the goal is to cover as much of an area as possible.
    The environment includes known and unknown obstacles that should be avoided.
    The input to the agent is a multi-scale map of the coverage and known obstacles.
    The maps are translated and rotated so that the mid-point of the maps
    correspond to the mowers position, and up is forward.
    The input also consists of a lidar sensor with distance measurements, which
    can sense the known and unknown obstacles.
    Multiple observations can be stacked.
    The actions that the agent can make are continuous throttle and steering.

    :param input_size: Size of the input maps [pixels], same resolution for all scales
    :param num_maps: Number of maps, i.e. scales
    :param scale_factor: Scale factor between each subsequent scale
    :param meters_per_pixel: World map resolution in meters per pixel
    :param min_size: Minimum size of free space for random maps [m]
    :param max_size: Maximum size of free space for random maps [m]
    :param stacks: Number of subsequent observations to stack
    :param step_size: Time per step [s]
    :param constant_lin_vel: Whether to use a constant linear velocity, i.e. agent only predicts steering
    :param max_lin_vel: Maximum linear velocity [m/s]
    :param max_ang_vel: Maximum angular velocity [rad/s]
    :param max_lin_acc: Maximum linear acceleration [m/s^2]
    :param max_ang_acc: Maximum angular acceleration [rad/s^2]
    :param action_delay: Delay before applying new action [s]
    :param steering_limits_lin_vel: Whether to suppress linear velocity based on steering angle
    :param mower_radius: Radius of cutting disc [m]
    :param lidar_rays: Number of lidar rays
    :param lidar_range: Range of the lidar sensor [m]
    :param lidar_fov: Lidar field of view [degrees]
    :param position_noise: Standard deviation for position noise [m]
    :param heading_noise: Standard deviation for heading noise [rad]
    :param lidar_noise: Standard deviation for lidar measurement noise [m]
    :param exploration: Whether coverage is defined by lidar detections (True), or robot extent (False)
    :param overlap_observation: Whether to use overlap (True) or binary coverage (False) in the map observation
    :param frontier_observation: Whether to use a frontier map as observation
    :param action_observations: Number of previous actions as observations
    :param eval: Eval mode, fixed eval maps, no training map progression, overrides next 6 parameters + some more
    :param p_use_known_obstacles: Probability per episode to randomly scatter known obstacles
    :param p_use_unknown_obstacles: Probability per episode to randomly scatter unknown obstacles
    :param p_use_floor_plans: Probability per episode generate random floor plans
    :param max_known_obstacles: Maximum number of known obstacles
    :param max_unknown_obstacles: Maximum number of unknown obstacles
    :param obstacle_radius: Radius of (circular) obstacles [m]
    :param all_unknown: Whether all obstacles and map geometry should be unknown, overrides padding parameters
    :param max_episode_steps: Maximum episode length
    :param max_non_new_steps: Maximum steps in a row which did not cover new ground
    :param collision_ends_episode: Whether a collisoin ends the current episode
    :param flip_when_stuck: Whether to turn the agent 180 degrees when it becomes stuck
    :param max_stuck_steps: Consequtive steps being stuck before flipping the agent
    :param start_level: Starting level for progressive training
    :param use_goal_time_in_levels: Whether to use time as a criteria for completing a level
    :param goal_coverage: Target coverage considered for task completion
    :param goal_coverage_reward: Reward for reaching the target coverage
    :param wall_collision_reward: Reward for wall collision
    :param obstacle_collision_reward: Reward for obstacle collision
    :param newly_visited_reward_scale: Reward multiplier when covering new ground
    :param newly_visited_reward_max: Maximum reward from covering new ground
    :param overlap_reward_scale: Reward multiplier for overlapping a previously covered area
    :param overlap_reward_max: Maximum overlap reward
    :param overlap_reward_always: Whether to always use overlap reward, or only when the coverage reward is 0
    :param local_tv_reward_scale: Reward multiplier for local (incremental) total variation
    :param local_tv_reward_max: Maximum reward from local (incremental) total variation
    :param global_tv_reward_scale: Reward multiplier for global total variation
    :param global_tv_reward_max: Maximum reward from global total variation
    :param use_known_obstacles_in_tv: Whether to include known obstacles when computing total variation
    :param use_unknown_obstacles_in_tv: Whether to include unknown obstacles when computing total variation
    :param frontier_reward_scale: Reward multiplier for frontier reward
    :param frontier_reward_max: Maximum frontier reward
    :param turn_reward_scale: Reward scale for turning penalty
    :param obstacle_dilation: Dilation kernel size around obstacles
    :param constant_reward: Constant reward at each time step
    :param constant_reward_always: Whether to always apply constant reward or only when no new space is covered
    :param truncation_reward_scale: Reward multiplier for episode truncation
    :param coverage_pad_value: Pad value for coverage maps outside environment borders (0 or 1)
    :param obstacle_pad_value: Pad value for obstacle maps outside environment borders (0 or 1)
    :param verbose: Whether to print reward/timing information
    :param metrics_dir: Where to store map and metrics after each episode (None: don't store)
    """

    def __init__(
        self,
        input_size = 32,
        num_maps = 4,
        scale_factor = 4,
        meters_per_pixel = 0.0375,
        min_size = None,
        max_size = None,
        stacks = 1,
        step_size = 0.5,
        constant_lin_vel = True,
        max_lin_vel = 0.26,
        max_ang_vel = 1.0,
        max_lin_acc = None,
        max_ang_acc = None,
        action_delay = 0,
        steering_limits_lin_vel = True,
        mower_radius = 0.15,
        lidar_rays = 24,
        lidar_range = 3.5,
        lidar_fov = 345,
        position_noise = 0.01,
        heading_noise = 0.05,
        lidar_noise = 0.05,
        exploration = False,
        overlap_observation = True,
        frontier_observation = True,
        action_observations = 0,
        eval = False,
        p_use_known_obstacles = 0.7,
        p_use_unknown_obstacles = 0.7,
        p_use_floor_plans = 0.7,
        max_known_obstacles = 100,
        max_unknown_obstacles = 100,
        obstacle_radius = 0.25,
        all_unknown = True,
        max_episode_steps = None,
        max_non_new_steps = 1000,
        collision_ends_episode = False,
        flip_when_stuck = False,
        max_stuck_steps = 5,
        start_level = 1,
        use_goal_time_in_levels = False,
        goal_coverage = 0.9,
        goal_coverage_reward = 0,
        wall_collision_reward = -10,
        obstacle_collision_reward = -10,
        newly_visited_reward_scale = 1,
        newly_visited_reward_max = 2,
        overlap_reward_scale = 0,
        overlap_reward_max = 5,
        overlap_reward_always = False,
        local_tv_reward_scale = 1,
        local_tv_reward_max = 5,
        global_tv_reward_scale = 0,
        global_tv_reward_max = 5,
        use_known_obstacles_in_tv = True,
        use_unknown_obstacles_in_tv = True,
        frontier_reward_scale = 0,
        frontier_reward_max = 5,
        turn_reward_scale = 0,
        obstacle_dilation = 9,
        constant_reward = -0.1,
        constant_reward_always = True,
        truncation_reward_scale = 0,
        coverage_pad_value = 0,
        obstacle_pad_value = 1,
        verbose = False,
        metrics_dir = None,
    ):
        super(MowerEnv, self).__init__()
        self.t1 = round(time.time() * 1000)

        # Environment parameters
        self.input_size = input_size
        self.num_maps = num_maps
        self.scale_factor = scale_factor
        self.meters_per_pixel = meters_per_pixel
        self.stacks = stacks
        self.step_size = step_size
        self.constant_lin_vel = constant_lin_vel
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.max_lin_acc = max_lin_acc
        self.max_ang_acc = max_ang_acc
        self.action_delay = action_delay
        self.steering_limits_lin_vel = steering_limits_lin_vel
        self.mower_radius = mower_radius
        self.lidar_rays = lidar_rays
        self.lidar_range = lidar_range
        self.lidar_fov = lidar_fov
        self.position_noise = position_noise
        self.heading_noise = heading_noise
        self.lidar_noise = lidar_noise
        self.exploration = exploration
        self.overlap_observation = overlap_observation
        self.frontier_observation = frontier_observation
        self.action_observations = action_observations
        self.eval = eval
        self.p_use_known_obstacles = p_use_known_obstacles
        self.p_use_unknown_obstacles = p_use_unknown_obstacles
        self.p_use_floor_plans = p_use_floor_plans
        self.max_known_obstacles = max_known_obstacles
        self.max_unknown_obstacles = max_unknown_obstacles
        self.obstacle_radius = obstacle_radius
        self.all_unknown = all_unknown
        self.max_episode_steps = max_episode_steps
        self.max_non_new_steps = max_non_new_steps
        self.collision_ends_episode = collision_ends_episode
        self.flip_when_stuck = flip_when_stuck
        self.max_stuck_steps = max_stuck_steps
        self.start_level = start_level
        self.use_goal_time_in_levels = use_goal_time_in_levels
        self.goal_coverage = goal_coverage
        self.goal_coverage_reward = goal_coverage_reward
        self.wall_collision_reward = wall_collision_reward
        self.obstacle_collision_reward = obstacle_collision_reward
        self.newly_visited_reward_scale = newly_visited_reward_scale
        self.newly_visited_reward_max = newly_visited_reward_max
        self.overlap_reward_scale = overlap_reward_scale
        self.overlap_reward_max = overlap_reward_max
        self.overlap_reward_always = overlap_reward_always
        self.local_tv_reward_scale = local_tv_reward_scale
        self.local_tv_reward_max = local_tv_reward_max
        self.global_tv_reward_scale = global_tv_reward_scale
        self.global_tv_reward_max = global_tv_reward_max
        self.use_known_obstacles_in_tv = use_known_obstacles_in_tv
        self.use_unknown_obstacles_in_tv = use_unknown_obstacles_in_tv
        self.frontier_reward_scale = frontier_reward_scale
        self.frontier_reward_max = frontier_reward_max
        self.turn_reward_scale = turn_reward_scale
        self.obstacle_dilation = obstacle_dilation
        self.constant_reward = constant_reward
        self.constant_reward_always = constant_reward_always
        self.truncation_reward_scale = truncation_reward_scale
        self.coverage_pad_value = coverage_pad_value
        self.obstacle_pad_value = obstacle_pad_value
        self.verbose = verbose
        self.metrics_dir = metrics_dir
        self.line_type = cv2.LINE_8

        # Derived parameters
        self.pixels_per_meter = 1 / meters_per_pixel
        self.ms_reach_p = self.input_size * self.scale_factor ** (self.num_maps - 1)
        self.ms_reach_m = self.ms_reach_p * self.meters_per_pixel
        if min_size is not None and max_size is not None:
            assert min_size <= max_size
        if min_size is not None:
            self.min_size_p = min_size * self.pixels_per_meter
        elif exploration:
            self.min_size_p = 256
        else:
            self.min_size_p = 64
        if max_size is not None:
            self.max_size_p = max_size * self.pixels_per_meter
            self.min_size_p = min(self.min_size_p, self.max_size_p)
        elif exploration:
            self.max_size_p = max(400, self.min_size_p)
        else:
            self.max_size_p = max(200, self.min_size_p)

        # Read map filenames
        if exploration:
            self.eval_maps = glob.glob('maps/eval_exploration*')
        else:
            self.eval_maps = glob.glob('maps/eval_mowing*')
        self.train_maps_0 = glob.glob('maps/train_0_*')
        self.train_maps_1 = glob.glob('maps/train_1_*')
        self.train_maps_2 = glob.glob('maps/train_2_*')
        self.train_maps_3 = glob.glob('maps/train_3_*')
        self.train_maps_4 = glob.glob('maps/train_4_*')
        self.train_maps_5 = glob.glob('maps/train_5_*')

        # Additional variables, set up level, reset maps etc.
        self.axes = None
        self.current_episode = 0
        self.level = start_level
        self.next_train_map = 0
        if not eval:
            self._set_level(self.level)

        # Action and observation spaces
        if constant_lin_vel:
            self.action_space = spaces.Box(low=-1, high=+1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1, high=+1, shape=(2,), dtype=np.float32)
        obs_shape = (self.stacks, self.num_maps, self.input_size, self.input_size)
        self.observation_space = spaces.Dict(
            coverage = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32),
            obstacles = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32),
            lidar = spaces.Box(low=0, high=1, shape=(self.stacks, self.lidar_rays), dtype=np.float32)
        )
        if frontier_observation:
            self.observation_space['frontier'] = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
        if self.action_observations > 0:
            action_obs_shape = (self.action_observations, self.action_space.shape[0])
            self.observation_space['action'] = spaces.Box(low=-1, high=+1, shape=action_obs_shape, dtype=np.float32)

    def reset(self):
        self.current_episode += 1
        self.elapsed_steps = 0
        self.non_new_steps = 0
        self.elapsed_time = 0
        self.num_collisions = 0
        self.stuck_steps = 0
        self.filename = None

        # Create a training environment
        if not self.eval:
            self._create_training_env()

        # Create an evaluation environment
        if self.eval:
            self.filename = self.eval_maps[(self.current_episode - 1) % len(self.eval_maps)]
            self._load_map(self.filename)

        # All unknown environment
        if self.all_unknown:
            self._set_all_unknown()

        # Randomize initial pose
        self._randomize_pose(self.mower_radius)

        # Initial lidar detections
        self.lidar_pts, pts_info, self.lidar_pts_max = self._compute_lidar_pts(self.position_m, self.heading)
        lidar_obs = self._compute_lidar_observation(self.lidar_pts, self.position_m)
        self._update_obstacles_from_lidar_pts(self.lidar_pts, pts_info)

        # Initialize maps, metrics etc.
        self._initialize_maps()
        self._initialize_observation(lidar_obs)
        self._initialize_metrics()

        # Return observation
        return self._get_stacked_observation()

    def step(self, action):
        self.t0 = round(time.time() * 1000)
        dt_other = str(self.t0 - self.t1)

        # Compute potential new pose
        assert self.step_size > self.action_delay
        if self.action_delay > 0:
            # apply old action
            lin_vel, ang_vel = self._compute_velocities(self._prev_action)
            new_position_m, new_heading = self._compute_new_pose(
                lin_vel, ang_vel, self.action_delay, self.position_m, self.heading)
            # apply new action
            lin_vel, ang_vel = self._compute_velocities(action)
            step_size = self.step_size - self.action_delay
            new_position_m, new_heading = self._compute_new_pose(
                lin_vel, ang_vel, step_size, new_position_m, new_heading)
        else:
            # apply new action only
            lin_vel, ang_vel = self._compute_velocities(action)
            new_position_m, new_heading = self._compute_new_pose(
                lin_vel, ang_vel, self.step_size, self.position_m, self.heading)
        self._prev_action = action

        # Check for collisions
        collided = False
        done = False
        reward_coll = 0
        if self._is_wall_collision(new_position_m, self.mower_radius):
            # out of bounds
            collided = True
            reward_coll = self.wall_collision_reward
        elif self._is_obstacle_collision(new_position_m):
            # collided with obstacle
            collided = True
            reward_coll = self.obstacle_collision_reward
        if collided:
            self._linvel = 0
            self._prev_action = [0] * self.action_space.shape[0]
            self.num_collisions += 1
            self.stuck_steps += 1
            if self.collision_ends_episode:
                done = True
        else:
            self.stuck_steps = 0

        # Flip the agent if stuck
        if self.flip_when_stuck and self.stuck_steps >= self.max_stuck_steps:
            self.stuck_steps = 0
            new_heading = (new_heading + math.pi) % (2 * math.pi)

        # Update pose
        self.old_heading = self.heading
        self.old_position_m = self.position_m.copy()
        self.heading = new_heading
        self.noisy_heading = np.random.normal(
            loc=self.heading, scale=self.heading_noise) % (2 * math.pi)
        if not collided:
            self.position_m = new_position_m
        else:
            new_position_m = self.position_m
        self.position_p = self.position_m * self.pixels_per_meter
        self.old_position_p = self.old_position_m * self.pixels_per_meter
        self.noisy_position_m = np.random.normal(
            loc=self.position_m, scale=self.position_noise)
        self.noisy_position_p = self.noisy_position_m * self.pixels_per_meter

        # Save local maps of the previous time step
        radius_m = self.mower_radius
        if self.exploration:
            radius_m = max(self.mower_radius, self.lidar_range)
        i1, i2, j1, j2 = self._get_local_neighborhood_indices(self.old_position_m, self.position_m, radius_m)
        self.local_overlap_old = self.overlap_map[i1:i2, j1:j2].copy()
        self.local_coverage_old = self.coverage_map[i1:i2, j1:j2].copy()
        self.local_known_obstacles_old = self.known_obstacle_map[i1:i2, j1:j2].copy()
        self.local_unknown_obstacles_old = self.unknown_obstacle_map[i1:i2, j1:j2].copy()

        # Compute lidar point cloud and observation
        self.old_lidar_pts = self.lidar_pts.copy()
        self.old_lidar_pts_max = self.lidar_pts_max.copy()
        dt_lidar = time.time() * 1000
        self.lidar_pts, pts_info, self.lidar_pts_max = self._compute_lidar_pts(self.position_m, self.heading)
        dt_lidar = round(time.time() * 1000 - dt_lidar)
        lidar_obs = self._compute_lidar_observation(self.lidar_pts, self.position_m)
        self._update_obstacles_from_lidar_pts(self.lidar_pts, pts_info)

        # Update the coverage and overlap maps
        local_coverage_diff = self._compute_coverage_diff(i1, i2, j1, j2)
        self.overlap_map[i1:i2, j1:j2] += local_coverage_diff
        self.coverage_map[i1:i2, j1:j2] = self.overlap_map[i1:i2, j1:j2].clip(max=1)

        # Update the frontier map
        d = max(1, 2 + self.obstacle_dilation // 2)
        ii1 = max(0, i1 - d); ii2 = min(self.size_p, i2 + d)
        jj1 = max(0, j1 - d); jj2 = min(self.size_p, j2 + d)
        local_frontier = self._compute_frontier_map(
            self.coverage_map[ii1:ii2, jj1:jj2], self.known_obstacle_map[ii1:ii2, jj1:jj2])
        self.frontier_map[i1:i2, j1:j2] = local_frontier[i1-ii1:i2-ii1, j1-jj1:j2-jj1]

        # Compute newly visited positions in this time step
        local_coverage_new = self.coverage_map[i1:i2, j1:j2]
        newly_visited = local_coverage_new > self.local_coverage_old
        newly_visited_sum = newly_visited.sum()

        # Update current coverage metrics
        all_obstacles = np.maximum(
            self.known_obstacle_map[i1:i2, j1:j2],
            self.unknown_obstacle_map[i1:i2, j1:j2])
        if self.obstacle_dilation > 1:
            kernel = np.ones((self.obstacle_dilation,)*2, dtype=float)
            all_obstacles[0, :] = 1; all_obstacles[-1, :] = 1
            all_obstacles[:, 0] = 1; all_obstacles[:, -1] = 1
            all_obstacles = cv2.dilate(all_obstacles, kernel, iterations=1)
        newly_visited_dilated = newly_visited.copy()
        newly_visited_dilated[all_obstacles > 0] = 0
        newly_visited_dilated_sum = newly_visited_dilated.sum()
        local_coverage_diff_dilated = local_coverage_diff.copy()
        local_coverage_diff_dilated[all_obstacles > 0] = 0
        local_coverage_diff_dilated_sum = local_coverage_diff_dilated.sum()
        self.coverage_in_pixels += newly_visited_dilated_sum
        self.coverage_in_m2 = self.coverage_in_pixels / (self.pixels_per_meter ** 2)
        self.coverage_in_percent = self.coverage_in_pixels / self.free_space_dilated
        self.overlap_in_pixels += local_coverage_diff_dilated_sum - newly_visited_dilated_sum
        self.overlap_in_m2 = self.overlap_in_pixels / (self.pixels_per_meter ** 2)
        if self.coverage_in_pixels != 0:
            self.overlap_in_percent = self.overlap_in_pixels / self.coverage_in_pixels
        else:
            self.overlap_in_percent = 0

        # Update current global total variation metric
        local_obstacles_old = None
        local_obstacles_new = None
        if self.use_known_obstacles_in_tv and self.use_unknown_obstacles_in_tv:
            local_obstacles_old = np.maximum(self.local_known_obstacles_old, self.local_unknown_obstacles_old)
            local_obstacles_new = np.maximum(self.known_obstacle_map[i1:i2, j1:j2], self.unknown_obstacle_map[i1:i2, j1:j2])
        elif self.use_known_obstacles_in_tv:
            local_obstacles_old = self.local_known_obstacles_old.copy()
            local_obstacles_new = self.known_obstacle_map[i1:i2, j1:j2].copy()
        elif self.use_unknown_obstacles_in_tv:
            local_obstacles_old = self.local_unknown_obstacles_old.copy()
            local_obstacles_new = self.unknown_obstacle_map[i1:i2, j1:j2].copy()
        if local_obstacles_new is not None and self.obstacle_dilation > 1:
            kernel = np.ones((self.obstacle_dilation,)*2, dtype=float)
            local_obstacles_old[0, :] = 1; local_obstacles_old[-1, :] = 1
            local_obstacles_old[:, 0] = 1; local_obstacles_old[:, -1] = 1
            local_obstacles_new[0, :] = 1; local_obstacles_new[-1, :] = 1
            local_obstacles_new[:, 0] = 1; local_obstacles_new[:, -1] = 1
            local_obstacles_old = cv2.dilate(local_obstacles_old, kernel, iterations=1)
            local_obstacles_new = cv2.dilate(local_obstacles_new, kernel, iterations=1)
        total_variation_old = utils.total_variation(self.local_coverage_old, local_obstacles_old)
        total_variation_new = utils.total_variation(local_coverage_new, local_obstacles_new)
        total_variation_diff = total_variation_new - total_variation_old
        self.global_total_variation += total_variation_diff

        # Reset non-new steps if we visited any new position
        if newly_visited_sum > 0:
            self.non_new_steps = 0

        # Update the observation
        n = self.elapsed_steps % self.stacks
        self.observation['lidar'][n] = lidar_obs
        if self.overlap_observation:
            cov_map = np.tanh(0.2 * self.overlap_map)
        else:
            cov_map = self.coverage_map
        self.observation['coverage'][n] = self._get_multi_scale_map(cov_map, pad_value=self.coverage_pad_value)
        self.observation['obstacles'][n] = self._get_multi_scale_map(self.known_obstacle_map, pad_value=self.obstacle_pad_value)
        if self.frontier_observation:
            self.observation['frontier'][n] = self._get_multi_scale_map(self.frontier_map, pad_value=0, ceil=True)
        if self.action_observations > 0:
            m = self.elapsed_steps % self.action_observations
            self.observation['action'][m] = action

        # Update the closest frontier distance
        old_frontier_distance = self.frontier_distance
        self.frontier_distance = self._compute_closest_frontier_distance(
            self.observation['frontier'][n])

        # Compute coverage-based rewards
        reward_area = 0
        reward_ovrlp = 0
        if not collided:

            # Normalization constant: maximum possible area covered per step
            # The maximum area corresponds to a rectangle based on the radius and maximum velocity
            width_m = 2 * self.mower_radius
            if self.exploration:
                width_m = 2 * self.lidar_range
            length_m = self.max_lin_vel * self.step_size
            max_newly_visited_sum = width_m * length_m * self.pixels_per_meter ** 2

            # Reward based on newly covered area
            reward_area = self.newly_visited_reward_scale * newly_visited_dilated_sum
            reward_area = reward_area / max_newly_visited_sum
            reward_area = min(reward_area, self.newly_visited_reward_max)

            # Reward based on overlap
            if self.overlap_reward_always or reward_area == 0:
                reward_ovrlp = (local_coverage_diff * self.local_overlap_old).sum()
                reward_ovrlp *= self.overlap_reward_scale / max_newly_visited_sum
                reward_ovrlp = min(reward_ovrlp, self.overlap_reward_max)
                reward_ovrlp = -reward_ovrlp

        # Reward based on total variation
        reward_tv = 0
        if not collided:

            # Local (incremental) total variation reward
            reward_itv = -total_variation_diff
            # normalize by the speed
            # to make the TV independent of pixel resolution and max speed etc.
            # (it is already independent of the radius)
            reward_itv *= self.meters_per_pixel / self.step_size / self.max_lin_vel
            # normalize such that TV of going straight forward at max speed = 1
            # the correct constant is probably 2 (2 sides of the mower = TV diff)
            # but 2.5 seemed more accurate due to the grid discretization
            reward_itv /= 2.5
            reward_itv *= self.local_tv_reward_scale
            reward_itv = np.sign(reward_itv) * min(abs(reward_itv), self.local_tv_reward_max)

            # Global total variation reward
            reward_gtv = -self.global_total_variation
            # normalize by the area
            reward_gtv /= math.sqrt(self.coverage_in_pixels)
            # normalize so that TV of a disk = 1
            # the correct constant is probably 2*sqrt(pi) (circumference over sqrt(area))
            # but 4 is more accurate due to the grid discretization
            reward_gtv /= 4
            reward_gtv *= self.global_tv_reward_scale
            reward_gtv = np.sign(reward_gtv) * min(abs(reward_gtv), self.global_tv_reward_max)

            # Final TV reward
            reward_tv = reward_itv + reward_gtv

        # Reward based on closest frontier
        reward_frontier = 0
        if not collided and newly_visited_sum == 0:
            reward_frontier = old_frontier_distance - self.frontier_distance
            reward_frontier *= self.frontier_reward_scale
            reward_frontier /= self.max_lin_vel * self.step_size
            reward_frontier = max(reward_frontier, -self.frontier_reward_max)
            reward_frontier = min(0, reward_frontier)

        # Turning penalty
        reward_turn = 0
        if not collided:
            reward_turn = -self.turn_reward_scale * abs(ang_vel / self.max_ang_vel)

        # Check if we reached the goal coverage
        reward_goal = 0
        if not done and self.coverage_in_percent >= self.goal_coverage:
            reward_goal = self.goal_coverage_reward
            done = True

            # Set the current map as completed
            if not self.eval:
                time_goal_reached = not self.use_goal_time_in_levels or self.elapsed_steps <= self.goal_steps
                if time_goal_reached:
                    if self.current_map is not None:
                        self.completed_maps[self.current_map] = True
                    else:
                        if self.use_floor_plans:
                            self.completed_floor_plan = True
                        if self.use_known_obstacles or self.use_unknown_obstacles:
                            self.completed_obstacles = True

                if np.all(self.completed_maps) and self.completed_floor_plan and self.completed_obstacles:
                    self.level += 1
                    self.next_train_map = 0
                    self._set_level(self.level)

        # Truncate episode if needed
        self.elapsed_steps += 1
        self.non_new_steps += 1
        self.elapsed_time = self.elapsed_steps * self.step_size
        info = {}
        reward_trunc = 0
        if self.max_episode_steps is not None:
            if not done and self.elapsed_steps >= self.max_episode_steps:
                reward_trunc = -self.truncation_reward_scale * (1 - self.coverage_in_percent)
                done = True
                info['TimeLimit.truncated'] = not done
        if self.max_non_new_steps is not None:
            if not done and self.non_new_steps >= self.max_non_new_steps:
                reward_trunc = -self.truncation_reward_scale * (1 - self.coverage_in_percent)
                done = True
                info['TimeLimit.non_new'] = True
        info['level'] = self.level

        # Compute observation and reward
        obs = self._get_stacked_observation()
        if self.constant_reward_always or newly_visited_sum == 0:
            reward_const = self.constant_reward
        else:
            reward_const = 0
        reward = reward_area + reward_ovrlp + reward_tv + reward_frontier + reward_turn + reward_coll + reward_goal + reward_trunc + reward_const

        # Log metrics
        self._log_metrics(done, action=action, reward=reward)

        # Print stuff
        if self.verbose:
            self.t1 = round(time.time() * 1000)
            dt_step = self.t1 - self.t0
            print('Step:', str(self.elapsed_steps) + ',',
                  'Coverage:' + str(round(100*self.coverage_in_percent, 2)).rjust(5) + ',',
                  'Reward:' + str(round(reward, 2)).rjust(5),
                  '(area:' + str(round(reward_area, 2)).rjust(4) + ',',
                  'TV:' + str(round(reward_tv, 2)).rjust(5) + ',',
                  'frontier:' + str(round(reward_frontier, 2)).rjust(5) + '),',
                  f'Time: {dt_step}/{dt_lidar}/{dt_other} ms step/lidar/etc')
        return obs, reward, done, info

    def render(self, mode):
        assert mode in ['full', 'limited', 'rgb_array']

        # Construct pyplot figure
        if self.axes is None:
            if mode == 'full':
                self.fig, self.axes = plt.subplot_mosaic('AABC;AAD.', constrained_layout=True)
                self.fig.set_size_inches(10, 5)
            elif mode == 'limited':
                self.fig, self.axes = plt.subplot_mosaic('A', constrained_layout=True)
                self.fig.set_size_inches(8, 8)
            else: # mode == 'rgb_array':
                self.fig, self.axes = plt.subplot_mosaic('A', constrained_layout=True)
                self.fig.set_size_inches(8, 8)
            for ax in self.axes:
                self.axes[ax].get_xaxis().set_visible(False)
                self.axes[ax].get_yaxis().set_visible(False)
        for ax in self.axes:
            self.axes[ax].clear()

        # Draw environment image, with coverage/obstacles/frontier/agent
        img = np.ones((self.size_p, self.size_p, 3), dtype=float)
        #overlap_tanh = np.tanh(0.5 * self.overlap_map)
        overlap_tanh = np.tanh(0.5 * self.coverage_map)
        img[:, :, 0] = np.clip(1 - 1.5*overlap_tanh, 0, 1)
        img[:, :, 1] = np.clip(2 - 1.5*overlap_tanh, 0, 1)
        img[:, :, 2] = np.clip(1 - 1.5*overlap_tanh, 0, 1)
        img[self.unknown_obstacle_map > 0] = 0.5
        img[self.known_obstacle_map > 0] = 0
        img[self.frontier_map > 0, 0] = 1
        img[self.frontier_map > 0, 1] = 0
        img[self.frontier_map > 0, 2] = 1
        cv2.circle(
            img,
            center=np.flip(self.position_p).astype(np.int32),
            radius=int(self.mower_radius * self.pixels_per_meter),
            color=[1, 0, 0],
            thickness=cv2.FILLED,
            lineType=self.line_type)
        self.axes['A'].imshow(np.flip(img.transpose(1, 0, 2), axis=0), interpolation='nearest')

        # Draw lidar detections
        for n in range(self.lidar_rays):
            self.axes['A'].plot(
                [self.pixels_per_meter * self.position_m[0] - 0.5, self.lidar_pts[n, 0]],
                [self.size_p - self.pixels_per_meter * self.position_m[1] - 0.5,
                self.size_p - 1 - self.lidar_pts[n, 1]], '-b', linewidth=0.5, zorder=10)

        # Draw an arrow for the heading
        self.axes['A'].arrow(
            self.pixels_per_meter * self.position_m[0] - 0.5,
            self.pixels_per_meter * (self.size_m - self.position_m[1]) - 0.5,
            self.pixels_per_meter * 2 * self.mower_radius * math.cos(self.heading),
            self.pixels_per_meter * 2 * self.mower_radius * -math.sin(self.heading),
            head_width = 1, head_length = 1, zorder=11)

        # Draw the path
        self.path.append(self.position_m)
        path = self.pixels_per_meter * np.array(self.path)
        self.axes['A'].plot(path[:, 0] - 0.5, self.size_p - path[:, 1] - 0.5, '-', color='yellow', zorder=9)
        self.axes['A'].set_xlim([-0.5, self.size_p - 0.5])
        self.axes['A'].set_ylim([self.size_p - 0.5, -0.5])

        # Return image if mode is rgb_array
        if mode == 'rgb_array':
            coverage = round(100 * self.coverage_in_percent)
            overlap = round(100 * self.overlap_in_percent)
            self.fig.canvas.draw()
            rgb_img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            rgb_img = rgb_img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,)).copy()
            rgb_img[:81, :100] = 255
            rgb_img[:81, 100] = 0; rgb_img[:81, 0] = 0
            rgb_img[81, :100] = 0; rgb_img[0, :100] = 0
            rgb_img = Image.fromarray(rgb_img)
            draw = ImageDraw.Draw(rgb_img)
            font = ImageFont.truetype('rlm/fonts/FreeMono.ttf', 16)
            color = (0, 0, 0)
            draw.text((2, 1), f'{self.elapsed_steps} steps', color, font=font)
            draw.text((2, 1+16), f'{round(self.elapsed_time, 1)} s', color, font=font)
            draw.text((2, 1+16*2), f'{coverage}% cover', color, font=font)
            draw.text((2, 1+16*3), f'{overlap}% over', color, font=font)
            draw.text((2, 1+16*4), f'{self.num_collisions} coll', color, font=font)
            rgb_img = np.array(rgb_img)
            return rgb_img

        # Show and return if mode is limited
        if mode == 'limited':
            plt.show(block=False)
            plt.pause(0.001)
            return

        # Create images of observed coverage/obstacle/frontier maps
        ob = self._get_latest_observation()
        coverage_ms_img = self._get_image_from_multi_scale_map(ob['coverage'])
        obstacle_ms_img = self._get_image_from_multi_scale_map(ob['obstacles'])
        coverage_img = np.ones((self.ms_reach_p, self.ms_reach_p, 3), dtype=float)
        obstacle_img = np.ones((self.ms_reach_p, self.ms_reach_p, 3), dtype=float)
        coverage_img[:, :, 0] = np.clip(1 - 1.5*coverage_ms_img, 0, 1)
        coverage_img[:, :, 1] = np.clip(2 - 1.5*coverage_ms_img, 0, 1)
        coverage_img[:, :, 2] = np.clip(1 - 1.5*coverage_ms_img, 0, 1)
        obstacle_img[:, :, 0] = np.clip(1 - obstacle_ms_img, 0, 1)
        obstacle_img[:, :, 1] = np.clip(1 - obstacle_ms_img, 0, 1)
        obstacle_img[:, :, 2] = np.clip(1 - obstacle_ms_img, 0, 1)
        if self.frontier_observation:
            frontier_ms_img = self._get_image_from_multi_scale_map(ob['frontier'])
            frontier_img = np.ones((self.ms_reach_p, self.ms_reach_p, 3), dtype=float)
            frontier_img[:, :, 1] = np.clip(1 - frontier_ms_img, 0, 1)

        # Draw observed maps
        self.axes['B'].imshow(np.flip(coverage_img.transpose(1, 0, 2), axis=0))
        self.axes['C'].imshow(np.flip(obstacle_img.transpose(1, 0, 2), axis=0))
        if self.frontier_observation:
            self.axes['D'].imshow(np.flip(frontier_img.transpose(1, 0, 2), axis=0))

        # Add some labels
        coverage = round(100 * self.coverage_in_percent)
        overlap = round(100 * self.overlap_in_percent)
        self.fig.suptitle(
            f'{self.elapsed_steps} steps' +
            f', {round(self.elapsed_time, 1)} s'
            f', {self.num_collisions} collisions' +
            f', {coverage}% coverage' +
            f', {overlap}% overlap', fontsize=16)
        labels = ['full map', 'coverage observation', 'obstacle observation']
        letters = ['A', 'B', 'C']
        if self.frontier_observation:
            labels += ['frontier observation']
            letters += ['D']
        trans = mtransforms.ScaledTranslation(10/72, -5/72, self.fig.dpi_scale_trans)
        for label, letter in zip(labels, letters):
            self.axes[letter].text(0.0, 1.0, label, transform=self.axes[letter].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
        plt.show(block=False)
        plt.pause(0.001)

    def close(self):
        plt.close('all')

    def _randomize_pose(self, mower_radius):
        """
        Randomizes a pose that is not in collision with obstacles or walls.
        Sets the following variables:
          self.heading
          self.noisy_heading
          self.position_m
          self.position_p
          self.noisy_position_m
          self.noisy_position_p
          self.orth_vec_old
          self._linvel
          self._angvel
          self._prev_action
        """
        self.heading = np.random.uniform(2 * math.pi)
        self.noisy_heading = np.random.normal(
            loc=self.heading, scale=self.heading_noise) % (2 * math.pi)
        self.position_m = np.random.uniform(size=2) * self.size_m
        # make sure we don't spawn in the wall or in an obstacle
        tries = 0
        while self._is_wall_collision(self.position_m, mower_radius) or \
              self._is_obstacle_collision(self.position_m):
            self.position_m = np.random.uniform(size=2) * self.size_m
            tries += 1
            assert tries < 1000, 'Could not initialize random position in 1000 tries, try reducing obstacle amount'
        self.position_p = self.position_m * self.pixels_per_meter
        self.noisy_position_m = np.random.normal(
            loc=self.position_m, scale=self.position_noise)
        self.noisy_position_p = self.noisy_position_m * self.pixels_per_meter
        self.orth_vec_old = None
        self._linvel = 0
        self._angvel = 0
        self._prev_action = [0] * self.action_space.shape[0]

    def _compute_velocities(self, action):
        """
        Computes the linear and angular velocities from the action
        """
        if self.constant_lin_vel:
            throttle = 1
            steering = action[0]
        else:
            throttle = action[0]
            steering = action[1]
        throttle = np.clip(throttle, -1, 1)
        steering = np.clip(steering, -1, 1)
        lin_vel = throttle
        if self.steering_limits_lin_vel:
            lin_vel *= (1 - abs(steering))
        lin_vel *= self.max_lin_vel
        if lin_vel < 0:
            lin_vel = lin_vel / 2
        ang_vel = steering
        ang_vel *= self.max_ang_vel
        return lin_vel, ang_vel

    def _compute_new_pose(self, lin_vel, ang_vel, step_size, old_position_m, old_heading):
        """
        Computes a new pose based on desired linear and angular velocities
        """

        # Compute position difference
        if self.max_lin_acc is None:
            self._linvel = lin_vel
        else:
            linvel_diff = lin_vel - self._linvel
            max_linvel_diff = self.max_lin_acc * step_size
            if abs(linvel_diff) > max_linvel_diff:
                self._linvel += np.sign(linvel_diff) * max_linvel_diff
            else:
                self._linvel = lin_vel
        pos_diff = self._linvel * step_size

        # Compute angle difference
        if self.max_ang_acc is None:
            self._angvel = ang_vel
        else:
            angvel_diff = ang_vel - self._angvel
            max_angvel_diff = self.max_ang_acc * step_size
            if abs(angvel_diff) > max_angvel_diff:
                self._angvel += np.sign(angvel_diff) * max_angvel_diff
            else:
                self._angvel = ang_vel
        ang_diff = self._angvel * step_size

        # Compute new pose
        new_heading = old_heading + ang_diff
        new_heading = new_heading % (2 * math.pi)
        inter_heading = old_heading + ang_diff / 2
        pos_vec = pos_diff * np.array([math.cos(inter_heading), math.sin(inter_heading)])
        new_position_m = old_position_m + pos_vec
        return new_position_m, new_heading

    def _is_wall_collision(self, position_m, radius_m):
        return position_m[0] <= radius_m or \
               position_m[1] <= radius_m or \
               position_m[0] >= self.size_m - radius_m or \
               position_m[1] >= self.size_m - radius_m

    def _is_obstacle_collision(self, position_m):
        i1, i2, j1, j2 = self._get_local_neighborhood_indices(position_m, position_m, self.mower_radius)
        local_known_obstacle_map = self.known_obstacle_map[i1:i2, j1:j2]
        local_unknown_obstacle_map = self.unknown_obstacle_map[i1:i2, j1:j2]
        local_all_obstacle_map = np.logical_or(local_known_obstacle_map, local_unknown_obstacle_map)
        local_position_map = np.zeros_like(local_known_obstacle_map)
        position_p = position_m * self.pixels_per_meter - [i1, j1]
        cv2.circle(
            local_position_map,
            center=(np.flip(position_p)).astype(np.int32),
            radius=int(self.mower_radius * self.pixels_per_meter),
            color=1,
            thickness=cv2.FILLED,
            lineType=self.line_type)
        return np.logical_and(local_position_map, local_all_obstacle_map).any()

    def _set_all_unknown(self):
        """
        Set all obstacles as unknown and set coverage/obstacle pad values to 0.
        """
        self.unknown_obstacle_map = np.maximum(self.known_obstacle_map, self.unknown_obstacle_map)
        self.unknown_obstacle_map[0, :] = 1; self.unknown_obstacle_map[-1, :] = 1
        self.unknown_obstacle_map[:, 0] = 1; self.unknown_obstacle_map[:, -1] = 1
        self.known_obstacle_map = np.zeros_like(self.known_obstacle_map)
        self.coverage_pad_value = 0
        self.obstacle_pad_value = 0

    def _initialize_maps(self):
        """
        Initialize coverage/overlap/frontier maps
        """
        self.coverage_map = np.zeros((self.size_p, self.size_p), dtype=float)
        if not self.exploration:
            cv2.circle(
                self.coverage_map,
                center=np.flip(self.position_p).astype(np.int32),
                radius=int(self.mower_radius * self.pixels_per_meter),
                color=1,
                thickness=cv2.FILLED,
                lineType=self.line_type)
        else:
            cv2.fillPoly(
                self.coverage_map,
                [np.fliplr(np.concatenate((self.lidar_pts, [self.position_p.astype(np.int32)])))],
                color=1)
            idxs = np.logical_or(self.known_obstacle_map, self.unknown_obstacle_map)
            self.coverage_map[idxs] = 0
        self.overlap_map = self.coverage_map.copy()
        self.frontier_map = self._compute_frontier_map(self.coverage_map, self.known_obstacle_map)

    def _initialize_observation(self, lidar_obs):
        self.observation = {}
        self.observation['lidar'] = np.tile(lidar_obs, (self.stacks, 1))
        if self.overlap_observation:
            cov_map = np.tanh(0.2 * self.overlap_map)
        else:
            cov_map = self.coverage_map
        self.observation['coverage'] = np.tile(self._get_multi_scale_map(cov_map, pad_value=self.coverage_pad_value), (self.stacks, 1, 1, 1))
        self.observation['obstacles'] = np.tile(self._get_multi_scale_map(self.known_obstacle_map, pad_value=self.obstacle_pad_value), (self.stacks, 1, 1, 1))
        if self.frontier_observation:
            self.observation['frontier'] = np.tile(self._get_multi_scale_map(self.frontier_map, pad_value=0, ceil=True), (self.stacks, 1, 1, 1))
        if self.action_observations > 0:
            self.observation['action'] = np.zeros(self.observation_space['action'].shape, dtype=np.float32)

    def _initialize_metrics(self):

        # Map of all obstacles
        all_obstacles = np.maximum(
            self.known_obstacle_map,
            self.unknown_obstacle_map)
        if self.obstacle_dilation > 1:
            kernel = np.ones((self.obstacle_dilation,)*2, dtype=float)
            all_obstacles[0, :] = 1; all_obstacles[-1, :] = 1
            all_obstacles[:, 0] = 1; all_obstacles[:, -1] = 1
            all_obstacles = cv2.dilate(all_obstacles, kernel, iterations=1)
        self.total_obstacle_pixels_dilated = all_obstacles.sum()
        self.free_space_dilated = self.size_p ** 2 - self.total_obstacle_pixels_dilated

        # Initialize coverage and overlap
        coverage = self.coverage_map.copy()
        coverage[all_obstacles > 0] = 0
        overlap = self.overlap_map.copy()
        overlap[all_obstacles > 0] = 0
        self.coverage_in_pixels = coverage.sum()
        self.coverage_in_m2 = self.coverage_in_pixels / (self.pixels_per_meter ** 2)
        self.coverage_in_percent = self.coverage_in_pixels / self.free_space_dilated
        self.overlap_in_pixels = (overlap - coverage).sum()
        self.overlap_in_m2 = self.overlap_in_pixels / (self.pixels_per_meter ** 2)
        if self.coverage_in_pixels != 0:
            self.overlap_in_percent = self.overlap_in_pixels / self.coverage_in_pixels
        else:
            self.overlap_in_percent = 0

        # Initialize global TV
        radius_m = self.mower_radius
        if self.exploration:
            radius_m = max(self.mower_radius, self.lidar_range)
        local_coverage = self._get_local_neighborhood(self.coverage_map, self.position_m, self.position_m, radius_m)
        local_obstacles = None
        if self.use_known_obstacles_in_tv and self.use_unknown_obstacles_in_tv:
            local_known_obstacles = self._get_local_neighborhood(self.known_obstacle_map, self.position_m, self.position_m, radius_m)
            local_unknown_obstacles = self._get_local_neighborhood(self.unknown_obstacle_map, self.position_m, self.position_m, radius_m)
            local_obstacles = np.maximum(local_known_obstacles, local_unknown_obstacles)
        elif self.use_known_obstacles_in_tv:
            local_obstacles = self._get_local_neighborhood(self.known_obstacle_map, self.position_m, self.position_m, radius_m).copy()
        elif self.use_unknown_obstacles_in_tv:
            local_obstacles = self._get_local_neighborhood(self.unknown_obstacle_map, self.position_m, self.position_m, radius_m).copy()
        if local_obstacles is not None and self.obstacle_dilation > 1:
            kernel = np.ones((self.obstacle_dilation,)*2, dtype=float)
            local_obstacles[0, :] = 1; local_obstacles[-1, :] = 1
            local_obstacles[:, 0] = 1; local_obstacles[:, -1] = 1
            local_obstacles = cv2.dilate(local_obstacles, kernel, iterations=1)
        self.global_total_variation = utils.total_variation(local_coverage, local_obstacles)

        # Keep track of the path taken (only updated/used when rendering)
        self.path = [self.position_m]

        # Log metrics
        self.metrics_logged = False
        self.xs = [self.position_m[0]]
        self.ys = [self.position_m[1]]
        self.headings = [self.heading]
        self.steps = [self.elapsed_steps]
        self.times = [self.elapsed_time]
        self.lengths = [0]
        self.turns = [0]
        self.coverages = [self.coverage_in_percent]
        self.overlaps = [self.overlap_in_percent]
        self.collisions = [self.num_collisions]
        self.actions = [None]
        self.rewards = [None]

        # Initialize closest frontier distance
        self.frontier_distance = self._compute_closest_frontier_distance(
            self.observation['frontier'][0])

    def _log_metrics(self, done, action=None, reward=None):
        if self.metrics_dir is None:
            return

        # Store data
        self.xs.append(self.position_m[0])
        self.ys.append(self.position_m[1])
        self.headings.append(self.heading)
        self.steps.append(self.elapsed_steps)
        self.times.append(self.elapsed_time)
        self.lengths.append(self.lengths[-1] + np.linalg.norm(self.position_m - self.old_position_m))
        heading_diff = abs(self.heading - self.old_heading)
        heading_diff = min(heading_diff, 2 * math.pi - heading_diff)
        assert heading_diff >= 0 and heading_diff <= 2 * math.pi
        self.turns.append(self.turns[-1] + heading_diff / (2 * math.pi))
        self.coverages.append(self.coverage_in_percent)
        self.overlaps.append(self.overlap_in_percent)
        self.collisions.append(self.num_collisions)
        self.actions.append(action)
        self.rewards.append(reward)

        # Save data
        if done and not self.metrics_logged:
            self.metrics_logged = True
            os.makedirs(self.metrics_dir, exist_ok=True)
            filename = 'eval' if self.eval else 'train'
            filename = filename + f'_metrics_{self.current_episode}.csv'
            f = open(os.path.join(self.metrics_dir, filename), 'w')
            f.write('x,y,heading,steps,time,length,turns,coverage,overlap,collisions,actions,rewards\n')
            for i in range(len(self.steps)):
                if self.actions[i] is None:
                    _action = ''
                else:
                    _action = str(self.actions[i][0])
                    for _a in self.actions[i][1:]:
                        _action += '/' + str(_a)
                _reward = '' if self.rewards[i] is None else self.rewards[i]
                f.write(
                    f'{self.xs[i]},' + \
                    f'{self.ys[i]},' + \
                    f'{self.headings[i]},' + \
                    f'{self.steps[i]},' + \
                    f'{self.times[i]},' + \
                    f'{self.lengths[i]},' + \
                    f'{self.turns[i]},' + \
                    f'{self.coverages[i]},' + \
                    f'{self.overlaps[i]},' + \
                    f'{self.collisions[i]},' + \
                    f'{_action},' + \
                    f'{_reward}\n'
                )
            f.close()
            # Save map
            filename = 'eval' if self.eval else 'train'
            filename = filename + f'_map_{self.current_episode}.png'
            img = 255 * np.ones_like(self.known_obstacle_map, dtype=np.uint8)
            img[self.known_obstacle_map > 0] = 0
            img[self.unknown_obstacle_map > 0] = 0
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(self.metrics_dir, filename), img)

    def _compute_coverage_diff(self, i1, i2, j1, j2):
        """
        Compute the region covered in the current step compared to the previous.
        Note that this includes regions that were covered earlier than the
        immediate preceding step for computing the overlap map.

        i1, i2, j1, j2: Local neighborhood indices
        """
        local_coverage_diff = np.zeros((i2 - i1, j2 - j1), dtype=float)

        # Compute coverage diff based on mower position
        if not self.exploration:

            # Currently covered disc
            cv2.circle(
                local_coverage_diff,
                center=np.flip(self.position_p - [i1, j1]).astype(np.int32),
                radius=int(self.mower_radius * self.pixels_per_meter),
                color=1,
                thickness=cv2.FILLED,
                lineType=self.line_type)

            # Rectangle covered by moving
            head_vec_new = self.position_p - self.old_position_p
            head_vec_len = np.sqrt(head_vec_new[0] ** 2 + head_vec_new[1] ** 2)
            if head_vec_len > 1e-5:
                orth_vec_new = np.array([-head_vec_new[1], head_vec_new[0]])
                orth_vec_new *= self.mower_radius * self.pixels_per_meter / head_vec_len
                cv2.fillConvexPoly(
                    local_coverage_diff,
                    points=np.array(
                        [np.flip(self.old_position_p - [i1, j1] + orth_vec_new),
                         np.flip(self.position_p - [i1, j1] + orth_vec_new),
                         np.flip(self.position_p - [i1, j1] - orth_vec_new),
                         np.flip(self.old_position_p - [i1, j1] - orth_vec_new)]).astype(np.int32),
                    color=1,
                    lineType=self.line_type)
            else:
                orth_vec_new = None

            # Remove region covered in the previous step
            cv2.circle(
                local_coverage_diff,
                center=np.flip(self.old_position_p - [i1, j1]).astype(np.int32),
                radius=int(self.mower_radius * self.pixels_per_meter),
                color=0,
                thickness=cv2.FILLED,
                lineType=self.line_type)
            if self.orth_vec_old is not None:
                cv2.fillConvexPoly(
                    local_coverage_diff,
                    points=np.array(
                        [np.flip(self.old_position_p - [i1, j1] + self.orth_vec_old),
                         np.flip(self.old_position_p - [i1, j1] + self.orth_vec_old - self.head_vec_old),
                         np.flip(self.old_position_p - [i1, j1] - self.orth_vec_old - self.head_vec_old),
                         np.flip(self.old_position_p - [i1, j1] - self.orth_vec_old)]).astype(np.int32),
                    color=0,
                    lineType=self.line_type)

            # Save heading/orthogonal vectors for next step
            self.head_vec_old = head_vec_new
            self.orth_vec_old = orth_vec_new

        # Compute coverage diff based on explored area
        if self.exploration:

            # Region currently in view
            cv2.fillPoly(
                local_coverage_diff,
                [np.fliplr(np.concatenate((self.lidar_pts, [self.position_p.astype(np.int32)])) - [i1, j1])],
                color=1)

            # Remove previously covered region
            cv2.fillPoly(
                local_coverage_diff,
                [np.fliplr(np.concatenate((self.old_lidar_pts, [self.old_position_p.astype(np.int32)])) - [i1, j1])],
                color=0)

            # Don't consider pixels within lidar max range of old position as overlappable
            # This is to get a resonable overlap measure when points pop in and out of view
            overlappable_pixels = np.ones_like(local_coverage_diff)
            cv2.fillPoly(
                overlappable_pixels,
                [np.fliplr(self.old_lidar_pts_max - [i1, j1])],
                color=0)
            overlappable_pixels[self.local_coverage_old == 0] = 1
            local_coverage_diff *= overlappable_pixels

        # Remove covered obstacles
        idxs = np.logical_or(
            self.known_obstacle_map[i1:i2, j1:j2],
            self.unknown_obstacle_map[i1:i2, j1:j2])
        local_coverage_diff[idxs] = 0
        return local_coverage_diff

    def _compute_frontier_map(self, coverage_map, obstacle_map):
        """
        Computes frontier points, i.e. points that are on the border between
        covered and free space.
        """
        coverage_map = coverage_map.copy()
        obstacle_map = obstacle_map.copy()
        if self.obstacle_dilation > 1:
            kernel = np.ones((self.obstacle_dilation,)*2, dtype=float)
            obstacle_map[0, :] = 1; obstacle_map[-1, :] = 1
            obstacle_map[:, 0] = 1; obstacle_map[:, -1] = 1
            obstacle_map = cv2.dilate(obstacle_map, kernel, iterations=1)
        coverage_map[obstacle_map > 0] = 0
        free_map = np.logical_not(coverage_map + obstacle_map)
        kernel = np.ones((3, 3), dtype=float)
        coverage_map = cv2.dilate(coverage_map, kernel, iterations=1)
        return np.logical_and(coverage_map, free_map).astype(float)
    
    def _compute_closest_frontier_distance(self, ms_frontier_map):
        """
        Approximates the distance to the nearest frontier point based on a
        multi-scale frontier map.
        """
        center = (self.input_size - 1) / 2
        xx, yy = np.meshgrid(range(self.input_size), range(self.input_size))
        dist_map = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        for i in range(self.num_maps):
            distances = dist_map[ms_frontier_map[i] > 0]
            if len(distances) > 0:
                min_distance = min(distances)
                min_distance *= self.meters_per_pixel
                min_distance *= self.scale_factor ** i
                return min_distance
        # If no frontier point is present: return the distance corresponding to
        # the corner point in the largest scale
        min_distance = self.input_size / np.sqrt(2)
        min_distance *= self.meters_per_pixel
        min_distance *= self.scale_factor ** (self.num_maps - 1)
        return min_distance

    def _compute_lidar_pts(self, position_m, heading):
        """
        Computes a lidar point cloud from a given pose.

        Parameters
        ----------
        position_m : position of lidar sensor [m]
        heading : heading of lidar sensor [rad]

        Returns
        -------
        lidar_pts : array of detected 2D points in absolute pixel coordinates
        pts_info : detection info
            0 = max range (no detection)
            1 = known obstacle
            2 = unknown obstacle
            3 = out of bounds
        lidar_pts_max : array of 2D points at max range for each ray
        """
        lidar_pts = np.zeros((self.lidar_rays, 2), dtype=np.int32)
        lidar_pts_max = np.zeros((self.lidar_rays, 2), dtype=np.int32)
        pts_info = np.zeros(self.lidar_rays, dtype=np.int32)
        samples = int(self.lidar_range * self.pixels_per_meter) # number of samples per ray
        position_p = position_m * self.pixels_per_meter
        for n, angle in enumerate(np.linspace(-self.lidar_fov/2, self.lidar_fov/2, num=self.lidar_rays)):
            ang = heading + angle * math.pi / 180
            search_vec = np.array([math.cos(ang), math.sin(ang)])
            for s in range(samples):
                offset_p = (s + 1) * search_vec
                pos_p = position_p + offset_p
                i = int(pos_p[0])
                j = int(pos_p[1])
                if i < 0 or i >= self.size_p or j < 0 or j >= self.size_p:
                    # lidar ray reaches beyond the area
                    pts_info[n] = 3
                    break
                if self.known_obstacle_map[i, j] > 0:
                    # lidar ray hits a known obstacle
                    pts_info[n] = 1
                    break
                if self.unknown_obstacle_map[i, j] > 0:
                    # lidar ray hits an unknown obstacle
                    pts_info[n] = 2
                    break
            # Store detection point
            lidar_pts[n] = [i, j]
            # Store max range point
            offset_p = samples * search_vec
            pos_p = position_p + offset_p
            lidar_pts_max[n] = [int(pos_p[0]), int(pos_p[1])]
        # Add max range points for the full 360 fov
        angle_diff = self.lidar_fov / (self.lidar_rays - 1)
        angle = -self.lidar_fov / 2 - angle_diff
        while angle > -180:
            ang = heading + angle * math.pi / 180
            search_vec = np.array([math.cos(ang), math.sin(ang)])
            offset_p = samples * search_vec
            pos_p = position_p + offset_p
            lidar_pts_max = np.concatenate(
                ([[int(pos_p[0]), int(pos_p[1])]], lidar_pts_max), axis=0)
            angle -= angle_diff
        angle = self.lidar_fov / 2 + angle_diff
        while angle < 180:
            ang = heading + angle * math.pi / 180
            search_vec = np.array([math.cos(ang), math.sin(ang)])
            offset_p = samples * search_vec
            pos_p = position_p + offset_p
            lidar_pts_max = np.concatenate(
                (lidar_pts_max, [[int(pos_p[0]), int(pos_p[1])]]), axis=0)
            angle += angle_diff
        return lidar_pts, pts_info, lidar_pts_max

    def _compute_lidar_observation(self, lidar_pts, position_m):
        """
        Computes normalized lidar distances from a given point cloud.

        Parameters
        ----------
        lidar_pts : array of 2D points in absolute pixel coordinates
        position_m : position of lidar sensor [m]

        Returns
        -------
        lidar_obs : array of lidar distances, normalized to [0, 1]
        """
        lidar_obs = np.ones(self.lidar_rays)
        for n in range(self.lidar_rays):
            offset_m = lidar_pts[n] / self.pixels_per_meter - position_m
            dist = math.sqrt(offset_m[0] ** 2 + offset_m[1] ** 2)
            dist = np.random.normal(loc=dist, scale=self.lidar_noise)
            lidar_obs[n] = np.clip(dist / self.lidar_range, 0, 1)
        return lidar_obs

    def _update_obstacles_from_lidar_pts(self, lidar_pts, pts_info):
        """
        Update known obstacle map based on detected unknown obstacles.
        """
        for n in range(self.lidar_rays):
            if pts_info[n] == 2:
                i = lidar_pts[n, 0]
                j = lidar_pts[n, 1]
                self.known_obstacle_map[i, j] = 1

    def _get_transform_matrix(self, scale):
        heading_degrees = self.noisy_heading * 180 / math.pi
        translation_matrix_1 = np.eye(3)
        translation_matrix_1[0, 2] = -self.noisy_position_p[1] / scale
        translation_matrix_1[1, 2] = -self.noisy_position_p[0] / scale
        rotation_matrix = np.eye(3)
        rotation_matrix[:2] = cv2.getRotationMatrix2D(
            center = (0, 0),
            angle = 90 - heading_degrees,
            scale = 1)
        translation_matrix_2 = np.eye(3)
        translation_matrix_2[0, 2] = self.input_size / 2
        translation_matrix_2[1, 2] = self.input_size / 2
        return translation_matrix_2 @ rotation_matrix @ translation_matrix_1

    def _get_relative_map(self, map, pad_value, scale=1, ceil=False, floor=False):
        assert not (ceil and floor), "_get_relative_map cannot do both ceil and floor"
        # note: since cv2.warAffine scales poorly with the input size, first
        # downsample to the correct resolution and then perform the warp
        # also: warpAffine ignores cv2.INTER_AREA, so using it to downsample by
        # using scaling in the transformation matrix does not work well, see:
        # https://stackoverflow.com/questions/57477478/opencv-warpaffine-ignores-flags-cv2-inter-area
        sc = min(scale, self.size_p)
        matrix = self._get_transform_matrix(sc)
        relative_map = cv2.resize(map, (int(0.5 + self.size_p / sc),)*2, interpolation=cv2.INTER_AREA)
        relative_map = cv2.warpAffine(relative_map, M=matrix[:2], dsize=(self.input_size,)*2, borderValue=pad_value, flags=cv2.INTER_AREA)
        if ceil:
            relative_map = np.ceil(relative_map)
        if floor:
            relative_map = np.floor(relative_map)
        return relative_map

    def _get_multi_scale_map(self, map, pad_value, ceil=False, floor=False):
        ms_map = np.zeros((self.num_maps, self.input_size, self.input_size))
        for n in range(self.num_maps):
            ms_map[n] = self._get_relative_map(map, pad_value, self.scale_factor**n, ceil, floor)
        return ms_map

    def _get_image_from_multi_scale_map(self, ms_map):
        img = np.zeros((self.ms_reach_p, self.ms_reach_p), dtype=float)
        for n in range(self.num_maps):
            size = self.ms_reach_p // (self.scale_factor ** n)
            map = cv2.resize(ms_map[self.num_maps-n-1], (size,)*2, interpolation=cv2.INTER_NEAREST)
            i1 = self.ms_reach_p // 2 - size // 2
            i2 = self.ms_reach_p // 2 + size // 2
            img[i1:i2, i1:i2] = map
        return img

    def _get_local_neighborhood_indices(self, pos1_m, pos2_m, radius_m):
        """
        Returns the local neighborhood indices to be used for cropping.
        """
        i1 = min(pos1_m[0], pos2_m[0]) - radius_m
        i2 = max(pos1_m[0], pos2_m[0]) + radius_m
        j1 = min(pos1_m[1], pos2_m[1]) - radius_m
        j2 = max(pos1_m[1], pos2_m[1]) + radius_m
        i1 = max(0, min(self.size_p, int(i1 * self.pixels_per_meter - 10)))
        i2 = max(0, min(self.size_p, int(i2 * self.pixels_per_meter + 10)))
        j1 = max(0, min(self.size_p, int(j1 * self.pixels_per_meter - 10)))
        j2 = max(0, min(self.size_p, int(j2 * self.pixels_per_meter + 10)))
        return i1, i2, j1, j2

    def _get_local_neighborhood(self, map, pos1_m, pos2_m, radius_m):
        """
        Returns the local neighborhood of a map as a crop.
        """
        i1, i2, j1, j2 = self._get_local_neighborhood_indices(pos1_m, pos2_m, radius_m)
        return map[i1:i2, j1:j2]

    def _get_stacked_observation(self):
        """
        Returns observations of the latest consecutive time steps, oldest first
        """
        ob = self.observation
        n = self.elapsed_steps % self.stacks
        observation = \
            {'coverage': np.concatenate((ob['coverage'][n:], ob['coverage'][:n]), axis=0),
             'obstacles': np.concatenate((ob['obstacles'][n:], ob['obstacles'][:n]), axis=0),
             'lidar': np.concatenate((ob['lidar'][n:], ob['lidar'][:n]), axis=0)}
        if self.frontier_observation:
            observation['frontier'] = np.concatenate((ob['frontier'][n:], ob['frontier'][:n]), axis=0)
        if self.action_observations > 0:
            m = self.elapsed_steps % self.action_observations
            observation['action'] = np.concatenate((ob['action'][m:], ob['action'][:m]), axis=0)
        return observation

    def _get_latest_observation(self):
        ob = self.observation
        n = (self.elapsed_steps - 1) % self.stacks
        observation = \
            {'coverage': ob['coverage'][n],
             'obstacles': ob['obstacles'][n],
             'lidar': ob['lidar'][n]}
        if self.frontier_observation:
            observation['frontier'] = ob['frontier'][n]
        if self.action_observations > 0:
            m = self.elapsed_steps % self.action_observations
            observation['action'] = np.concatenate((ob['action'][m:], ob['action'][:m]), axis=0)
        return observation

    def _load_map(self, filename):
        img = cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE)
        img = np.fliplr(img.transpose(1, 0))
        assert len(img.shape) == 2
        assert img.shape[0] == img.shape[1]
        assert set(img.flatten()).issubset([0, 128, 255]), 'Invalid map image'
        self.size_p = img.shape[0]
        self.size_m = self.size_p / self.pixels_per_meter
        self.known_obstacle_map = np.zeros((self.size_p, self.size_p), dtype=float)
        self.unknown_obstacle_map = np.zeros((self.size_p, self.size_p), dtype=float)
        self.known_obstacle_map[img == 0] = 1
        self.unknown_obstacle_map[img == 128] = 1

    def _randomize_floor_plan(self):
        min_room_size_p = int(10 * self.mower_radius * self.pixels_per_meter)
        max_room_size_p = int(32 * self.mower_radius * self.pixels_per_meter)
        min_wall_thickness_p = 2
        max_wall_thickness_p = int(2 * self.mower_radius * self.pixels_per_meter)
        min_gap = int(4 * self.mower_radius * self.pixels_per_meter)
        max_gap = int(8 * self.mower_radius * self.pixels_per_meter)
        if self.size_p > 2 * min_room_size_p:
            room_size_p = random.randint(min_room_size_p, max_room_size_p)
            num_walls = max(1, int(self.size_p / room_size_p) - 1)
            room_size_p = int(self.size_p / (num_walls + 1))
            wall_thickness_p = random.randint(min_wall_thickness_p, max_wall_thickness_p)
            vertical_stop = random.uniform(0, 1) < 0.5
            for n in range(num_walls):
                i1 = room_size_p * (n + 1) - wall_thickness_p // 2
                i2 = room_size_p * (n + 1) + wall_thickness_p
                if random.uniform(0, 1) < 0.9:
                    # place vertical wall
                    self.known_obstacle_map[i1:i2, :] = 1
                if random.uniform(0, 1) < 0.9:
                    # place horizontal wall
                    self.known_obstacle_map[:, i1:i2] = 1
                stop_placed = False
                for m in range(num_walls + 1):
                    # open gaps in walls
                    g_min = min_gap
                    g_max = min(max_gap, room_size_p - 2 * wall_thickness_p)
                    j_min = room_size_p * m + wall_thickness_p
                    j_max = room_size_p * (m + 1) - wall_thickness_p
                    p_stop = 1 / (num_walls + 1 - m)
                    place_stop = False
                    if not stop_placed and random.uniform(0, 1) < p_stop:
                        place_stop = True
                        stop_placed = True
                    if not vertical_stop or not place_stop:
                        # open gap in vertical wall
                        gap = random.randint(g_min, g_max)
                        j1 = random.randint(j_min, j_max - gap)
                        j2 = j1 + gap
                        self.known_obstacle_map[i1:i2, j1:j2] = 0
                    if vertical_stop or not place_stop:
                        # open gap in horizontal wall
                        gap = random.randint(g_min, g_max)
                        j1 = random.randint(j_min, j_max - gap)
                        j2 = j1 + gap
                        self.known_obstacle_map[j1:j2, i1:i2] = 0

    def _randomize_circular_obstacles(self):
        known_pos = np.random.uniform(size=(self.max_known_obstacles, 2))
        unknown_pos = np.random.uniform(size=(self.max_unknown_obstacles, 2))
        radius = 2 * self.mower_radius + self.obstacle_radius
        assert self.size_m > 4 * radius
        for n in range(max(self.max_known_obstacles, self.max_unknown_obstacles)):
            # alternate placing known/unknown obstacles
            if self.use_known_obstacles and n < self.max_known_obstacles:
                pos_m = 2 * radius + known_pos[n] * (self.size_m - 4 * radius)
                local_known = self._get_local_neighborhood(
                    self.known_obstacle_map, pos_m, pos_m, radius)
                local_unknown = self._get_local_neighborhood(
                    self.unknown_obstacle_map, pos_m, pos_m, radius)
                if local_known.sum() == 0 and local_unknown.sum() == 0:
                    cv2.circle(
                        self.known_obstacle_map,
                        center=(np.flip(pos_m * self.pixels_per_meter)).astype(np.int32),
                        radius=int(self.obstacle_radius * self.pixels_per_meter),
                        color=1,
                        thickness=cv2.FILLED,
                        lineType=self.line_type)
            if self.use_unknown_obstacles and n < self.max_unknown_obstacles:
                pos_m = 2 * radius + unknown_pos[n] * (self.size_m - 4 * radius)
                local_known = self._get_local_neighborhood(
                    self.known_obstacle_map, pos_m, pos_m, radius)
                local_unknown = self._get_local_neighborhood(
                    self.unknown_obstacle_map, pos_m, pos_m, radius)
                if local_known.sum() == 0 and local_unknown.sum() == 0:
                    cv2.circle(
                        self.unknown_obstacle_map,
                        center=(np.flip(pos_m * self.pixels_per_meter)).astype(np.int32),
                        radius=int(self.obstacle_radius * self.pixels_per_meter),
                        color=1,
                        thickness=cv2.FILLED,
                        lineType=self.line_type)

    def _create_training_env(self):
        """
        Creates a training environment either by loading a fixed training map
        or by randomizing a floor plan with circular obstacles.
        """
        self.current_map = None
        self.use_floor_plans = False
        self.use_known_obstacles = False
        self.use_unknown_obstacles = False

        # Load fixed training map
        if not self.use_randomized_envs or random.uniform(0, 1) < 0.5:
            self.current_map = self.next_train_map
            self.next_train_map = (self.next_train_map + 1) % len(self.train_maps)
            self.filename = self.train_maps[self.current_map]
            self._load_map(self.filename)

        # Generate random map
        else:
            self.size_p = random.randint(self.min_size_p, self.max_size_p)
            self.size_m = self.size_p / self.pixels_per_meter
            self.known_obstacle_map = np.zeros((self.size_p, self.size_p), dtype=float)
            self.unknown_obstacle_map = np.zeros((self.size_p, self.size_p), dtype=float)
            self.use_floor_plans = random.uniform(0, 1) < self.p_use_floor_plans
            if self.use_floor_plans:
                self._randomize_floor_plan()
            self.use_known_obstacles = self.max_known_obstacles > 0 and random.uniform(0, 1) < self.p_use_known_obstacles
            self.use_unknown_obstacles = self.max_unknown_obstacles > 0 and random.uniform(0, 1) < self.p_use_unknown_obstacles
            if self.use_known_obstacles or self.use_unknown_obstacles:
                self._randomize_circular_obstacles()

    def _set_level(self, level):
        """
        Sets the followng variables based on the specified level:
          - self.train_maps
          - self.goal_coverage
          - self.goal_steps
          - self.use_randomized_envs
          - self.min_size_p
          - self.max_size_p
          - self.completed_maps
          - self.completed_floor_plan
          - self.completed_obstacles
        """
        self.goal_steps = 2000
        if self.exploration:
            if level == 1:
                self.goal_coverage = 0.9
                self.use_randomized_envs = False
                self.train_maps = \
                    self.train_maps_1 + \
                    self.train_maps_2
            elif level == 2:
                self.goal_coverage = 0.9
                self.use_randomized_envs = False
                self.train_maps = \
                    self.train_maps_1 + \
                    self.train_maps_2 + \
                    self.train_maps_4
            elif level == 3:
                self.goal_coverage = 0.95
                self.use_randomized_envs = False
                self.train_maps = \
                    self.train_maps_1 + \
                    self.train_maps_2 + \
                    self.train_maps_4
            elif level == 4:
                self.goal_coverage = 0.97
                self.use_randomized_envs = False
                self.train_maps = \
                    self.train_maps_1 + \
                    self.train_maps_2 + \
                    self.train_maps_4
            elif level == 5:
                self.goal_coverage = 0.99
                self.use_randomized_envs = False
                self.train_maps = \
                    self.train_maps_1 + \
                    self.train_maps_2 + \
                    self.train_maps_4
            elif level == 6:
                self.goal_coverage = 0.99
                self.use_randomized_envs = False
                self.train_maps = \
                    self.train_maps_1 + \
                    self.train_maps_2 + \
                    self.train_maps_3 + \
                    self.train_maps_4
            elif level == 7:
                self.goal_coverage = 0.99
                self.use_randomized_envs = True
                self.min_size_p = 256
                self.max_size_p = 320
                self.train_maps = \
                    self.train_maps_1 + \
                    self.train_maps_2 + \
                    self.train_maps_3 + \
                    self.train_maps_4
            elif level == 8:
                self.goal_coverage = 0.99
                self.use_randomized_envs = True
                self.min_size_p = 256
                self.max_size_p = 400
                self.train_maps = \
                    self.train_maps_1 + \
                    self.train_maps_2 + \
                    self.train_maps_3 + \
                    self.train_maps_4
            else:
                self.goal_coverage = 0.99
                self.use_randomized_envs = True
                self.min_size_p = 256
                self.max_size_p = 400
                self.train_maps = \
                    self.train_maps_1 + \
                    self.train_maps_2 + \
                    self.train_maps_3 + \
                    self.train_maps_4 + \
                    self.train_maps_5
        else:
            if level == 1:
                self.goal_coverage = 0.9
                self.use_randomized_envs = False
                self.train_maps = self.train_maps_0
            elif level == 2:
                self.goal_coverage = 0.9
                self.use_randomized_envs = False
                self.train_maps = \
                    self.train_maps_0 + \
                    self.train_maps_1
            elif level == 3:
                self.goal_coverage = 0.95
                self.use_randomized_envs = False
                self.train_maps = \
                    self.train_maps_0 + \
                    self.train_maps_1
            elif level == 4:
                self.goal_coverage = 0.95
                self.use_randomized_envs = False
                self.train_maps = \
                    self.train_maps_0 + \
                    self.train_maps_1 + \
                    self.train_maps_2
            elif level == 5:
                self.goal_coverage = 0.97
                self.use_randomized_envs = False
                self.train_maps = \
                    self.train_maps_0 + \
                    self.train_maps_1 + \
                    self.train_maps_2
            elif level == 6:
                self.goal_coverage = 0.99
                self.use_randomized_envs = False
                self.train_maps = \
                    self.train_maps_0 + \
                    self.train_maps_1 + \
                    self.train_maps_2
            elif level == 7:
                self.goal_coverage = 0.99
                self.use_randomized_envs = False
                self.train_maps = \
                    self.train_maps_0 + \
                    self.train_maps_1 + \
                    self.train_maps_2 + \
                    self.train_maps_3
            else:
                self.goal_coverage = 0.99
                self.use_randomized_envs = True
                self.train_maps = \
                    self.train_maps_0 + \
                    self.train_maps_1 + \
                    self.train_maps_2 + \
                    self.train_maps_3

        # Keep track of which maps have been completed
        self.completed_maps = [False]*len(self.train_maps)
        self.completed_floor_plan = True
        self.completed_obstacles = True
        if self.use_randomized_envs:
            if self.p_use_floor_plans > 0:
                self.completed_floor_plan = False
            else:
                self.completed_floor_plan = True
            use_known_obstacles = self.max_known_obstacles > 0 and self.p_use_known_obstacles > 0
            use_unknown_obstacles = self.max_unknown_obstacles > 0 and self.p_use_unknown_obstacles > 0
            if use_known_obstacles or use_unknown_obstacles:
                self.completed_obstacles = False
            else:
                self.completed_obstacles = True

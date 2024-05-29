import cv2
import math
import numpy as np
from rlm import utils
from rlm.mower_env import MowerEnv
from rlm.random_mower import RandomMower
from rlm.spiral_mower import SpiralMower


def test_min_max_sizes():
    """
    Test that the env.min_size_p and env.max_size_p are correct.
    """
    env = MowerEnv(min_size=50, meters_per_pixel=1)
    assert env.min_size_p == 50
    assert env.max_size_p >= 50
    env = MowerEnv(min_size=1000, meters_per_pixel=1)
    assert env.min_size_p == 1000
    assert env.max_size_p >= 50
    env = MowerEnv(max_size=50, meters_per_pixel=1)
    assert env.min_size_p <= 50
    assert env.max_size_p == 50
    env = MowerEnv(max_size=1000, meters_per_pixel=1)
    assert env.min_size_p <= 1000
    assert env.max_size_p == 1000
    env = MowerEnv(min_size=50, max_size=1000, meters_per_pixel=1)
    assert env.min_size_p == 50
    assert env.max_size_p == 1000
    env = MowerEnv(min_size=50, max_size=1000, meters_per_pixel=0.5)
    assert env.min_size_p == 100
    assert env.max_size_p == 2000

def test_all_unknown():
    """
    Test all_unknown parameter.
    """
    env = MowerEnv(all_unknown=True)
    env.reset()
    assert env.known_obstacle_map.sum() <= env.lidar_rays
    env = MowerEnv(all_unknown=True, exploration=True)
    env.reset()
    assert env.known_obstacle_map.sum() <= env.lidar_rays

def test_is_wall_collision():
    """
    Test for MowerEnv._is_wall_collision
    """
    r = 0.15
    envs = [MowerEnv(mower_radius=r),
            MowerEnv(mower_radius=r, eval=True),
            MowerEnv(mower_radius=r, exploration=True),
            MowerEnv(mower_radius=r, exploration=True, eval=True)]
    for env in envs:
        env.reset()
        s = env.size_m
        assert env._is_wall_collision([-1, -1], r)
        assert env._is_wall_collision([s+1, s+1], r)
        assert env._is_wall_collision([r/2]*2, r)
        assert env._is_wall_collision([2*r, s], 1)
        assert env._is_wall_collision([s, 2*r], 1)
        assert env._is_wall_collision([s - r/2]*2, r)
        assert not env._is_wall_collision([2*r]*2, r)
        assert not env._is_wall_collision([s/2]*2, r)
        assert not env._is_wall_collision([s - 2*r]*2, r)

def test_is_obstacle_collision():
    """
    Test for MowerEnv._is_obstacle_collision
    """
    env = MowerEnv(meters_per_pixel=0.1, mower_radius=0.3)
    env.reset()
    env.known_obstacle_map = np.zeros_like(env.known_obstacle_map)
    env.unknown_obstacle_map = np.zeros_like(env.unknown_obstacle_map)
    env.known_obstacle_map[10, 10] = 1
    assert env._is_obstacle_collision(np.array([1, 1]))
    assert env._is_obstacle_collision(np.array([1.1, 1.1]))
    assert not env._is_obstacle_collision(np.array([1.5, 1.5]))
    env.known_obstacle_map = np.zeros_like(env.known_obstacle_map)
    env.unknown_obstacle_map = np.zeros_like(env.unknown_obstacle_map)
    env.unknown_obstacle_map[10, 10] = 1
    assert env._is_obstacle_collision(np.array([1, 1]))
    assert env._is_obstacle_collision(np.array([1.1, 1.1]))
    assert not env._is_obstacle_collision(np.array([1.5, 1.5]))

def test_compute_frontier_map():
    """
    Test for MowerEnv._compute_frontier_map
    """
    # Test case 1
    env = MowerEnv(obstacle_dilation=0)
    coverage_map = np.zeros((8, 8))
    obstacle_map = np.zeros((8, 8))
    frontier_map = env._compute_frontier_map(coverage_map, obstacle_map)
    assert frontier_map.sum() == 0
    # Test case 2
    env = MowerEnv(obstacle_dilation=0)
    coverage_map = np.zeros((8, 8))
    obstacle_map = np.zeros((8, 8))
    coverage_map[2:4, 2:4] = [[1,1],[1,0]]
    obstacle_map[3:5, 3:5] = 1
    frontier_map = env._compute_frontier_map(coverage_map, obstacle_map)
    frontier_map_gt = np.zeros((8, 8))
    frontier_map_gt[1:5, 1:5] = [[1,1,1,1],[1,0,0,1],[1,0,0,0],[1,1,0,0]]
    assert (frontier_map == frontier_map_gt).all()
    # Test case 3
    env = MowerEnv(obstacle_dilation=5)
    coverage_map = np.zeros((16, 16))
    obstacle_map = np.zeros((16, 16))
    coverage_map[2:5, 2:5] = 1
    obstacle_map[6:9, 6:9] = 1
    frontier_map = env._compute_frontier_map(coverage_map, obstacle_map)
    frontier_map_gt = np.zeros((16, 16))
    frontier_map_gt[3, 5] = 1
    frontier_map_gt[5, 3] = 1
    assert (frontier_map == frontier_map_gt).all()
    # Test case 4 (consistency over time)
    steps = 100
    kwargs = [
        {'exploration': False, 'obstacle_dilation': 0},
        {'exploration': False, 'obstacle_dilation': 3},
        {'exploration': False, 'obstacle_dilation': 7},
        {'exploration': True, 'obstacle_dilation': 0},
        {'exploration': True, 'obstacle_dilation': 3},
        {'exploration': True, 'obstacle_dilation': 7}]
    mowers = [
        RandomMower,
        RandomMower,
        RandomMower,
        RandomMower,
        RandomMower,
        RandomMower]
    for kwarg, mower in zip(kwargs, mowers):
        env = MowerEnv(**kwarg)
        model = mower(env)
        obs = env.reset()
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done:
                break
        assert env.obstacle_dilation == kwarg['obstacle_dilation']
        frontier_map = env._compute_frontier_map(env.coverage_map, env.known_obstacle_map)
        assert (env.frontier_map == frontier_map).all()
        assert (env.frontier_map * env.coverage_map).sum() == 0
        env.close()

def test_compute_closest_frontier_distance():
    """
    Test for MowerEnv._compute_closest_frontier_distance
    """
    env = MowerEnv()
    # Test case 1
    env.input_size = 4
    env.num_maps = 2
    env.meters_per_pixel = 0.5
    env.scale_factor = 2
    ms_frontier_map = np.zeros((env.num_maps, env.input_size, env.input_size))
    dist = env._compute_closest_frontier_distance(ms_frontier_map)
    assert abs(dist - 2 * np.sqrt(2)) < 1e-8
    ms_frontier_map[1, -1, -1] = 1
    dist = env._compute_closest_frontier_distance(ms_frontier_map)
    assert abs(dist - 1.5 * np.sqrt(2)) < 1e-8
    ms_frontier_map[0, 0, 1] = 1
    dist = env._compute_closest_frontier_distance(ms_frontier_map)
    assert abs(dist - np.sqrt(10) / 4) < 1e-8
    # Test case 2
    env.input_size = 32
    env.num_maps = 4
    env.meters_per_pixel = 0.0375
    env.scale_factor = 4
    ms_frontier_map = np.zeros((env.num_maps, env.input_size, env.input_size))
    dist = env._compute_closest_frontier_distance(ms_frontier_map)
    assert abs(dist - 38.4 * np.sqrt(2)) < 1e-8
    ms_frontier_map[3, -1, -1] = 1
    dist = env._compute_closest_frontier_distance(ms_frontier_map)
    assert abs(dist - 37.2 * np.sqrt(2)) < 1e-8
    ms_frontier_map[2, 0, 0] = 1
    dist = env._compute_closest_frontier_distance(ms_frontier_map)
    assert abs(dist - 9.3 * np.sqrt(2)) < 1e-8
    ms_frontier_map[2, 0, 1] = 1
    dist = env._compute_closest_frontier_distance(ms_frontier_map)
    assert abs(dist - 0.3 * np.sqrt(1802)) < 1e-8
    ms_frontier_map[2, 0, 2] = 1
    dist = env._compute_closest_frontier_distance(ms_frontier_map)
    assert abs(dist - 0.3 * np.sqrt(1690)) < 1e-8
    ms_frontier_map[1, 16, 16] = 1
    dist = env._compute_closest_frontier_distance(ms_frontier_map)
    assert abs(dist - 0.15 / np.sqrt(2)) < 1e-8
    ms_frontier_map[0, 0, 0] = 1
    dist = env._compute_closest_frontier_distance(ms_frontier_map)
    assert abs(dist - 0.58125 * np.sqrt(2)) < 1e-8
    ms_frontier_map[0, 16, 16] = 1
    dist = env._compute_closest_frontier_distance(ms_frontier_map)
    assert abs(dist - 0.0375 / np.sqrt(2)) < 1e-8

def test_compute_lidar_pts():
    """
    Test for MowerEnv._compute_lidar_pts
    """
    # Test case 1
    env = MowerEnv(lidar_rays=3, lidar_range=2, lidar_fov=180, meters_per_pixel=1)
    env.known_obstacle_map = np.zeros((8, 8))
    env.unknown_obstacle_map = np.zeros((8, 8))
    env.size_p = 8
    env.heading = math.pi / 2
    env.position_p = np.array([4, 4])
    env.position_m = env.position_p * env.meters_per_pixel
    lidar_pts, pts_info, _ = env._compute_lidar_pts(env.position_m, env.heading)
    assert np.allclose(lidar_pts, [[6,4],[4,6],[2,4]], rtol=1e-05, atol=1e-08)
    assert (pts_info == 0).all()
    # Test case 2
    env = MowerEnv(lidar_rays=7, lidar_range=1, lidar_fov=270, meters_per_pixel=0.1)
    env.known_obstacle_map = np.zeros((8, 8))
    env.unknown_obstacle_map = np.zeros((8, 8))
    env.size_p = 8
    env.heading = math.pi / 4
    env.position_p = np.array([4, 4])
    env.position_m = env.position_p * env.meters_per_pixel
    lidar_pts, pts_info, _ = env._compute_lidar_pts(env.position_m, env.heading)
    dists = lidar_pts - env.position_p
    dists = np.sqrt(dists[:, 0]**2 + dists[:, 1]**2)
    assert (dists[0] < dists[1::2]).all()
    assert (dists[2] < dists[1::2]).all()
    assert (dists[4] < dists[1::2]).all()
    assert (dists[6] < dists[1::2]).all()
    assert (pts_info == 3).all()
    # Test case 3
    env = MowerEnv(lidar_rays=3, lidar_range=2, lidar_fov=180, meters_per_pixel=0.1)
    env.known_obstacle_map = np.zeros((8, 8))
    env.unknown_obstacle_map = np.zeros((8, 8))
    env.known_obstacle_map[:, 3] = 1
    env.unknown_obstacle_map[:, 5] = 1
    env.size_p = 8
    env.heading = 0
    env.position_p = np.array([4, 4])
    env.position_m = env.position_p * env.meters_per_pixel
    lidar_pts, pts_info, _ = env._compute_lidar_pts(env.position_m, env.heading)
    assert np.allclose(lidar_pts[0], [4,3], rtol=1e-05, atol=1e-08)
    assert np.allclose(lidar_pts[1], [8,4], rtol=1e-05, atol=1e-08)
    assert np.allclose(lidar_pts[2], [4,5], rtol=1e-05, atol=1e-08)
    assert np.allclose(pts_info, [1,3,2], rtol=1e-05, atol=1e-08)

def test_compute_lidar_observation():
    """
    Test for MowerEnv._compute_lidar_observation
    """
    # Test case 1
    env = MowerEnv(lidar_rays=4, lidar_range=1, meters_per_pixel=1, lidar_noise=0)
    lidar_pts = np.array([[4,3], [3,4], [2,3], [3,2]])
    position_m = np.array([3,3])
    lidar_obs = env._compute_lidar_observation(lidar_pts, position_m)
    assert (lidar_obs == 1).all()
    # Test case 2
    env = MowerEnv(lidar_rays=3, lidar_range=10, meters_per_pixel=0.1, lidar_noise=0)
    position_p = np.array([1000, 1000])
    position_m = position_p * env.meters_per_pixel
    obs1 = 0.1
    obs2 = 0.5
    obs3 = 1
    vec1 = np.array([1, 1])
    vec2 = np.array([0.5, 1])
    vec3 = np.array([49, 70])
    vec1 = obs1 * env.lidar_range * vec1 / np.linalg.norm(vec1)
    vec2 = obs2 * env.lidar_range * vec2 / np.linalg.norm(vec2)
    vec3 = obs3 * env.lidar_range * vec3 / np.linalg.norm(vec3)
    lidar_pts = np.array([vec1, vec2, vec3]) + position_m
    lidar_pts *= env.pixels_per_meter
    lidar_obs = env._compute_lidar_observation(lidar_pts, position_m)
    assert np.allclose(lidar_obs, [obs1, obs2, obs3], rtol=1e-05, atol=1e-08)

def test_get_transform_matrix():
    """
    Test for MowerEnv._get_transform_matrix
    """
    def get_matrix(position_p, heading, scale, size):
        p0 = position_p[0]; p1 = position_p[1]
        theta = math.pi / 2 - heading
        s = scale; w = size
        return np.array([
            [math.cos(theta),
             math.sin(theta),
             -p1/s*math.cos(theta) - p0/s*math.sin(theta) + w/2],
            [-math.sin(theta),
             math.cos(theta),
             p1/s*math.sin(theta) - p0/s*math.cos(theta) + w/2],
            [0, 0, 1]
        ])
    envs = [MowerEnv(input_size=32),
            MowerEnv(input_size=16),
            MowerEnv(input_size=8),
            MowerEnv(input_size=4)]
    headings = [math.pi / 2, math.pi / 4, 1, 2]
    positions = [0.5, 0, 1, 0.8]
    for env, heading, position in zip(envs, headings, positions):
        env.reset()
        env.noisy_heading = heading
        env.noisy_position_p = [position * env.size_p] * 2
        matrix1 = env._get_transform_matrix(scale=1)
        matrix2 = env._get_transform_matrix(scale=2)
        matrix3 = env._get_transform_matrix(scale=0.5)
        matrix1_gt = get_matrix(env.noisy_position_p, env.noisy_heading, 1, env.input_size)
        matrix2_gt = get_matrix(env.noisy_position_p, env.noisy_heading, 2, env.input_size)
        matrix3_gt = get_matrix(env.noisy_position_p, env.noisy_heading, 0.5, env.input_size)
        assert np.allclose(matrix1, matrix1_gt, rtol=1e-05, atol=1e-08)
        assert np.allclose(matrix2, matrix2_gt, rtol=1e-05, atol=1e-08)
        assert np.allclose(matrix3, matrix3_gt, rtol=1e-05, atol=1e-08)

def test_get_relative_map():
    """
    Test for MowerEnv._get_relative_map
    """
    # Test 1
    env = MowerEnv(input_size=4, mower_radius=1, meters_per_pixel=1)
    map = np.zeros((8, 8))
    map[3, 3] = 1
    env.size_p = 8
    env.noisy_heading = math.pi / 2
    env.noisy_position_p = [4, 4]
    rel1 = env._get_relative_map(map, pad_value=1, scale=0.5, ceil=False, floor=False)
    rel2 = env._get_relative_map(map, pad_value=1, scale=1, ceil=False, floor=False)
    rel3 = env._get_relative_map(map, pad_value=1, scale=2, ceil=False, floor=False)
    rel4 = env._get_relative_map(map, pad_value=1, scale=4, ceil=False, floor=False)
    assert np.allclose(rel1, [[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]], rtol=1e-05, atol=1e-08)
    assert np.allclose(rel2, [[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]], rtol=1e-05, atol=1e-08)
    assert np.allclose(rel3, [[0,0,0,0],[0,1/4,0,0],[0,0,0,0],[0,0,0,0]], rtol=1e-05, atol=1e-08)
    assert np.allclose(rel4, [[1,1,1,1],[1,1/16,0,1],[1,0,0,1],[1,1,1,1]], rtol=1e-05, atol=1e-08)
    # Test 2
    env = MowerEnv(input_size=16)
    map = np.ones((8, 8))
    map[4:, :4] = 0
    env.size_p = 8
    env.noisy_heading = math.pi / 4
    env.noisy_position_p = [4, 4]
    rel = env._get_relative_map(map, pad_value=0, scale=0.5)
    assert rel[0, 8] == 1
    assert rel[8, 0] == 1
    assert rel[8, 15] == 1
    assert rel[15, 8] == 0
    rel = env._get_relative_map(map, pad_value=0, scale=2)
    assert rel[0, 8] == 0
    assert rel[8, 0] == 0
    assert rel[8, 15] == 0
    assert rel[15, 8] == 0

def test_get_multi_scale_map():
    """
    Test for MowerEnv._get_multi_scale_map
    """
    # Test case 1
    env = MowerEnv(input_size=4, num_maps=3, scale_factor=2)
    map = np.zeros((8, 8))
    map[2:4, 2:4] = 1
    env.size_p = 8
    env.noisy_heading = math.pi / 2
    env.noisy_position_p = [4, 4]
    map = env._get_multi_scale_map(map, pad_value=0)
    assert np.allclose(map, [[[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]],
                             [[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]],
                             [[0,0,0,0],[0,1/4,0,0],[0,0,0,0],[0,0,0,0]]], rtol=1e-05, atol=1e-08)
    # Test case 2
    env = MowerEnv(input_size=8, num_maps=3, scale_factor=2)
    map = np.zeros((32, 32))
    map[12:16, 12:16] = 1
    env.size_p = 32
    env.noisy_heading = math.pi / 2
    env.noisy_position_p = [16, 16]
    map = env._get_multi_scale_map(map, pad_value=0)
    assert np.allclose(map[0,0:4,0:4], [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]], rtol=1e-05, atol=1e-08)
    assert np.allclose(map[1,0:4,0:4], [[0,0,0,0],[0,0,0,0],[0,0,1,1],[0,0,1,1]], rtol=1e-05, atol=1e-08)
    assert np.allclose(map[2,0:4,0:4], [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]], rtol=1e-05, atol=1e-08)
    # Test case 3
    env = MowerEnv(input_size=8, num_maps=2, scale_factor=2)
    map = np.zeros((16, 16))
    map[4:, 4:] = 1
    env.size_p = 16
    env.noisy_heading = math.pi / 2
    env.noisy_position_p = [0, 0]
    map = env._get_multi_scale_map(map, pad_value=1)
    assert np.allclose(map[0,0:4,0:4], 1, rtol=1e-05, atol=1e-08)
    assert np.allclose(map[0,4:8,4:8], 0, rtol=1e-05, atol=1e-08)
    assert np.allclose(map[1,0:4,0:4], 1, rtol=1e-05, atol=1e-08)
    assert np.allclose(map[1,4:8,4:8], [[0,0,0,0],[0,0,0,0],[0,0,1,1],[0,0,1,1]], rtol=1e-05, atol=1e-08)

def test_get_image_from_multi_scale_map():
    """
    Test for MowerEnv._get_image_from_multi_scale_map
    """
    env = MowerEnv(input_size=4, num_maps=3, scale_factor=2)
    ms_map = np.array([[[1,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
                       [[0,1,0,0], [0,1,0,0], [0,0,0,1], [0,0,0,1]],
                       [[1,1,0,0], [1,1,0,0], [0,0,1,0], [0,0,0,1]]])
    img_true = np.array([[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                         [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                         [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                         [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                         [1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0],
                         [1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0],
                         [1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0],
                         [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
                         [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
                         [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
                         [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]])
    img = env._get_image_from_multi_scale_map(ms_map)
    assert np.allclose(img, img_true, rtol=1e-05, atol=1e-08)

def test_get_local_neighborhood_indices():
    """
    Test for MowerEnv._get_local_neighborhood_indices
    """
    # Test case 1
    env = MowerEnv(meters_per_pixel=1)
    env.size_p = 100
    i1, i2, j1, j2 = env._get_local_neighborhood_indices([50,50], [50,50], 10)
    assert i1 > 0 and i1 < 50-10
    assert j1 > 0 and j1 < 50-10
    assert i2 > 50+10 and i2 < 100
    assert j2 > 50+10 and j2 < 100
    # Test case 2
    env = MowerEnv(meters_per_pixel=1)
    env.size_p = 100
    i1, i2, j1, j2 = env._get_local_neighborhood_indices([50,50], [50,50], 1000)
    assert i1 == 0
    assert j1 == 0
    assert i2 == 100
    assert j2 == 100
    # Test case 3
    env = MowerEnv(meters_per_pixel=0.1)
    env.size_p = 200
    i1, i2, j1, j2 = env._get_local_neighborhood_indices([5,5], [10,10], 1)
    assert i1 > 0 and i1 < 50-10
    assert j1 > 0 and j1 < 50-10
    assert i2 > 100+10 and i2 < 200
    assert j2 > 100+10 and j2 < 200

def test_get_local_neighborhood():
    """
    Test for MowerEnv._get_local_neighborhood
    """
    # Test case 1
    env = MowerEnv(meters_per_pixel=1)
    global_map = np.zeros((100, 100))
    global_map[40:60, 40:60] = 1
    env.size_p = 100
    local_map = env._get_local_neighborhood(global_map, [50,50], [50,50], 10)
    assert (np.array(local_map.shape) > 20).all()
    assert (np.array(local_map.shape) < 100).all()
    assert local_map.sum() == 20*20
    # Test case 2
    env = MowerEnv(meters_per_pixel=1)
    global_map = np.zeros((100, 100))
    global_map[40:60, 40:60] = 1
    env.size_p = 100
    local_map = env._get_local_neighborhood(global_map, [50,50], [50,50], 1000)
    assert (np.array(local_map.shape) == 100).all()
    assert local_map.sum() == 20*20
    # Test case 3
    env = MowerEnv(meters_per_pixel=0.1)
    global_map = np.zeros((200, 200))
    global_map[10:150, 10:150] = 1
    env.size_p = 200
    local_map = env._get_local_neighborhood(global_map, [5,5], [10,10], 1)
    assert (np.array(local_map.shape) > 50+20).all()
    assert (np.array(local_map.shape) < 200).all()
    assert local_map.sum() == local_map.size

def test_get_stacked_observation():
    """
    Test for MowerEnv._get_stacked_observation
    """
    # Test case 1
    env = MowerEnv(frontier_observation=False)
    env.observation = \
        {'coverage': [1,2,3,4,5],
         'obstacles': [2,3,4,5,1],
         'lidar': [5,4,3,2,1]}
    env.elapsed_steps = 0
    env.stacks = 5
    obs = env._get_stacked_observation()
    assert len(obs.keys()) == 3
    assert 'coverage' in obs.keys()
    assert 'obstacles' in obs.keys()
    assert 'lidar' in obs.keys()
    assert (obs['coverage'] == [1, 2, 3, 4, 5]).all()
    assert (obs['obstacles'] == [2, 3, 4, 5, 1]).all()
    assert (obs['lidar'] == [5, 4, 3, 2, 1]).all()
    env.elapsed_steps = 2
    obs = env._get_stacked_observation()
    assert (obs['coverage'] == [3, 4, 5, 1, 2]).all()
    assert (obs['obstacles'] == [4, 5, 1, 2, 3]).all()
    assert (obs['lidar'] == [3, 2, 1, 5, 4]).all()
    env.elapsed_steps = 4
    obs = env._get_stacked_observation()
    assert (obs['coverage'] == [5, 1, 2, 3, 4]).all()
    assert (obs['obstacles'] == [1, 2, 3, 4, 5]).all()
    assert (obs['lidar'] == [1, 5, 4, 3, 2]).all()
    env.elapsed_steps = 5
    obs = env._get_stacked_observation()
    assert (obs['coverage'] == [1, 2, 3, 4, 5]).all()
    assert (obs['obstacles'] == [2, 3, 4, 5, 1]).all()
    assert (obs['lidar'] == [5, 4, 3, 2, 1]).all()
    # Test case 2
    env = MowerEnv(frontier_observation=True)
    env.observation = \
        {'coverage': [1,2,3,4,5],
         'obstacles': [2,3,4,5,1],
         'frontier': [6,7,8,9,10],
         'lidar': [5,4,3,2,1]}
    env.elapsed_steps = 0
    env.stacks = 5
    obs = env._get_stacked_observation()
    assert len(obs.keys()) == 4
    assert 'coverage' in obs.keys()
    assert 'obstacles' in obs.keys()
    assert 'frontier' in obs.keys()
    assert 'lidar' in obs.keys()
    assert (obs['coverage'] == [1, 2, 3, 4, 5]).all()
    assert (obs['obstacles'] == [2, 3, 4, 5, 1]).all()
    assert (obs['frontier'] == [6, 7, 8, 9, 10]).all()
    assert (obs['lidar'] == [5, 4, 3, 2, 1]).all()
    env.elapsed_steps = 2
    obs = env._get_stacked_observation()
    assert (obs['coverage'] == [3, 4, 5, 1, 2]).all()
    assert (obs['obstacles'] == [4, 5, 1, 2, 3]).all()
    assert (obs['frontier'] == [8, 9, 10, 6, 7]).all()
    assert (obs['lidar'] == [3, 2, 1, 5, 4]).all()
    env.elapsed_steps = 4
    obs = env._get_stacked_observation()
    assert (obs['coverage'] == [5, 1, 2, 3, 4]).all()
    assert (obs['obstacles'] == [1, 2, 3, 4, 5]).all()
    assert (obs['frontier'] == [10, 6, 7, 8, 9]).all()
    assert (obs['lidar'] == [1, 5, 4, 3, 2]).all()
    env.elapsed_steps = 5
    obs = env._get_stacked_observation()
    assert (obs['coverage'] == [1, 2, 3, 4, 5]).all()
    assert (obs['obstacles'] == [2, 3, 4, 5, 1]).all()
    assert (obs['frontier'] == [6, 7, 8, 9, 10]).all()
    assert (obs['lidar'] == [5, 4, 3, 2, 1]).all()

def test_get_latest_observation():
    """
    Test for MowerEnv._get_latest_observation
    """
    # Test case 1
    env = MowerEnv(frontier_observation=False)
    env.observation = \
        {'coverage': [1,2,3,4,5],
         'obstacles': [2,3,4,5,1],
         'lidar': [5,4,3,2,1]}
    env.elapsed_steps = 0
    env.stacks = 5
    obs = env._get_latest_observation()
    assert len(obs.keys()) == 3
    assert 'coverage' in obs.keys()
    assert 'obstacles' in obs.keys()
    assert 'lidar' in obs.keys()
    assert obs['coverage'] == 5
    assert obs['obstacles'] == 1
    assert obs['lidar'] == 1
    env.elapsed_steps = 2
    obs = env._get_latest_observation()
    assert obs['coverage'] == 2
    assert obs['obstacles'] == 3
    assert obs['lidar'] == 4
    env.elapsed_steps = 4
    obs = env._get_latest_observation()
    assert obs['coverage'] == 4
    assert obs['obstacles'] == 5
    assert obs['lidar'] == 2
    env.elapsed_steps = 5
    obs = env._get_latest_observation()
    assert obs['coverage'] == 5
    assert obs['obstacles'] == 1
    assert obs['lidar'] == 1
    # Test case 2
    env = MowerEnv(frontier_observation=True)
    env.observation = \
        {'coverage': [1,2,3,4,5],
         'obstacles': [2,3,4,5,1],
         'frontier': [6,7,8,9,10],
         'lidar': [5,4,3,2,1]}
    env.elapsed_steps = 0
    env.stacks = 5
    obs = env._get_latest_observation()
    assert len(obs.keys()) == 4
    assert 'coverage' in obs.keys()
    assert 'obstacles' in obs.keys()
    assert 'frontier' in obs.keys()
    assert 'lidar' in obs.keys()
    assert obs['coverage'] == 5
    assert obs['obstacles'] == 1
    assert obs['frontier'] == 10
    assert obs['lidar'] == 1
    env.elapsed_steps = 2
    obs = env._get_latest_observation()
    assert obs['coverage'] == 2
    assert obs['obstacles'] == 3
    assert obs['frontier'] == 7
    assert obs['lidar'] == 4
    env.elapsed_steps = 4
    obs = env._get_latest_observation()
    assert obs['coverage'] == 4
    assert obs['obstacles'] == 5
    assert obs['frontier'] == 9
    assert obs['lidar'] == 2
    env.elapsed_steps = 5
    obs = env._get_latest_observation()
    assert obs['coverage'] == 5
    assert obs['obstacles'] == 1
    assert obs['frontier'] == 10
    assert obs['lidar'] == 1

def test_initial_observation():
    """
    Test that the initial observation is consistent.
    """
    env = MowerEnv(stacks=3)
    env.reset()
    latest_obs = env._get_latest_observation()
    stacked_obs = env._get_stacked_observation()
    assert len(stacked_obs['coverage']) == 3
    assert len(stacked_obs['obstacles']) == 3
    assert np.allclose(latest_obs['coverage'], stacked_obs['coverage'][0], rtol=1e-05, atol=1e-08)
    assert np.allclose(latest_obs['coverage'], stacked_obs['coverage'][1], rtol=1e-05, atol=1e-08)
    assert np.allclose(latest_obs['coverage'], stacked_obs['coverage'][2], rtol=1e-05, atol=1e-08)
    assert np.allclose(latest_obs['obstacles'], stacked_obs['obstacles'][0], rtol=1e-05, atol=1e-08)
    assert np.allclose(latest_obs['obstacles'], stacked_obs['obstacles'][1], rtol=1e-05, atol=1e-08)
    assert np.allclose(latest_obs['obstacles'], stacked_obs['obstacles'][2], rtol=1e-05, atol=1e-08)

def test_load_map():
    """
    Test for MowerEnv._load_map
    """
    # Test case 1
    env = MowerEnv(meters_per_pixel=1)
    env._load_map('tests/test_map_1.png')
    known_obstacle_map = np.zeros((20, 20))
    unknown_obstacle_map = np.zeros((20, 20))
    known_obstacle_map[:10, :10] = 1
    unknown_obstacle_map[10:, 10:] = 1
    known_obstacle_map = np.fliplr(known_obstacle_map.transpose(1, 0))
    unknown_obstacle_map = np.fliplr(unknown_obstacle_map.transpose(1, 0))
    assert env.size_p == 20
    assert env.size_m == 20
    assert (env.known_obstacle_map == known_obstacle_map).all()
    assert (env.unknown_obstacle_map == unknown_obstacle_map).all()
    # Test case 2
    env = MowerEnv(meters_per_pixel=0.2)
    env._load_map('tests/test_map_2.png')
    known_obstacle_map = np.zeros((20, 20))
    unknown_obstacle_map = np.zeros((20, 20))
    known_obstacle_map[0:5, 0:5] = 1
    known_obstacle_map[10:15, 10:15] = 1
    known_obstacle_map[0:5, 15:20] = 1
    known_obstacle_map[5:10, 10:15] = 1
    unknown_obstacle_map[5:10, 5:10] = 1
    unknown_obstacle_map[15:20, 15:20] = 1
    unknown_obstacle_map[10:15, 5:10] = 1
    unknown_obstacle_map[15:20, 0:5] = 1
    known_obstacle_map = np.fliplr(known_obstacle_map.transpose(1, 0))
    unknown_obstacle_map = np.fliplr(unknown_obstacle_map.transpose(1, 0))
    assert env.size_p == 20
    assert env.size_m == 4
    assert (env.known_obstacle_map == known_obstacle_map).all()
    assert (env.unknown_obstacle_map == unknown_obstacle_map).all()

def test_set_level():
    """
    Test that MowerEnv._set_level resets map completion
    """
    envs = [
        MowerEnv(exploration=False),
        MowerEnv(exploration=True)]
    for env in envs:
        for level in range(10):
            env._set_level(level)
            assert not np.array(env.completed_maps).any()
            if env.use_randomized_envs and env.p_use_floor_plans > 0:
                assert not env.completed_floor_plan
            else:
                assert env.completed_floor_plan
            use_known_obstacles = env.max_known_obstacles > 0 and env.p_use_known_obstacles > 0
            use_unknown_obstacles = env.max_unknown_obstacles > 0 and env.p_use_unknown_obstacles > 0
            use_obstacles = use_known_obstacles or use_unknown_obstacles
            if env.use_randomized_envs and use_obstacles:
                assert not env.completed_obstacles
            else:
                assert env.completed_obstacles

def test_coverage():
    """
    Test for coverage metrics and maps.
    """
    # Test case 1
    steps = 100
    kwargs = [
        {'exploration': False, 'obstacle_dilation': 0},
        {'exploration': False, 'obstacle_dilation': 1},
        {'exploration': True, 'obstacle_dilation': 0},
        {'exploration': True, 'obstacle_dilation': 1},
        {'exploration': False, 'obstacle_dilation': 5},
        {'exploration': True, 'obstacle_dilation': 3},
        {'exploration': False, 'obstacle_dilation': 0, 'step_size': 5},
        {'exploration': True, 'obstacle_dilation': 0, 'step_size': 5}]
    mowers = [
        RandomMower,
        SpiralMower,
        RandomMower,
        SpiralMower,
        RandomMower,
        SpiralMower,
        RandomMower,
        RandomMower]
    for kwarg, mower in zip(kwargs, mowers):
        env = MowerEnv(**kwarg)
        model = mower(env)
        obs = env.reset()
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done:
                break
        assert env.obstacle_dilation == kwarg['obstacle_dilation']
        all_obstacles = np.maximum(
            env.known_obstacle_map,
            env.unknown_obstacle_map)
        if env.obstacle_dilation > 1:
            kernel = np.ones((env.obstacle_dilation,)*2, dtype=float)
            all_obstacles[0, :] = 1; all_obstacles[-1, :] = 1
            all_obstacles[:, 0] = 1; all_obstacles[:, -1] = 1
            all_obstacles = cv2.dilate(all_obstacles, kernel, iterations=1)
        total_obstacle_pixels = all_obstacles.sum()
        free_space = env.size_p ** 2 - total_obstacle_pixels
        coverage = env.coverage_map.copy()
        coverage[all_obstacles > 0] = 0
        overlap = env.overlap_map.copy()
        overlap[all_obstacles > 0] = 0
        coverage_in_pixels = coverage.sum()
        overlap_in_pixels = (overlap - coverage).sum()
        assert env.coverage_in_pixels == coverage_in_pixels
        assert env.coverage_in_m2 == coverage_in_pixels / (env.pixels_per_meter ** 2)
        assert env.coverage_in_percent == coverage_in_pixels / free_space
        assert env.overlap_in_pixels == overlap_in_pixels
        assert env.overlap_in_m2 == overlap_in_pixels / (env.pixels_per_meter ** 2)
        if coverage_in_pixels != 0:
            assert env.overlap_in_percent == overlap_in_pixels / coverage_in_pixels
        else:
            assert env.overlap_in_percent == 0
        assert (env.coverage_map[env.known_obstacle_map > 0] == 0).all()
        assert (env.coverage_map[env.unknown_obstacle_map > 0] == 0).all()
        assert (env.overlap_map[env.known_obstacle_map > 0] == 0).all()
        assert (env.overlap_map[env.unknown_obstacle_map > 0] == 0).all()
        assert (env.coverage_map <= env.overlap_map).all()
        env.close()
    # Test case 2
    for _ in range(100):
        env = MowerEnv(meters_per_pixel=1, exploration=False)
        env.reset()
        if env.coverage_in_pixels == 0:
            assert env.coverage_in_m2 == 0
            assert env.coverage_in_percent == 0
            assert env.overlap_in_pixels == 0
            assert env.overlap_in_m2 == 0
            assert env.overlap_in_percent == 0
        env.close()

def test_total_variation():
    """
    Test that the global total variation metric is consistent over time.
    """
    steps = 100
    kwargs = [
        {'exploration': False, 'use_known_obstacles_in_tv': True, 'use_unknown_obstacles_in_tv': True, 'obstacle_dilation': 5},
        {'exploration': True, 'use_known_obstacles_in_tv': True, 'use_unknown_obstacles_in_tv': True, 'obstacle_dilation': 5},
        {'exploration': False, 'use_known_obstacles_in_tv': False, 'use_unknown_obstacles_in_tv': False, 'obstacle_dilation': 5},
        {'exploration': False, 'use_known_obstacles_in_tv': True, 'use_unknown_obstacles_in_tv': False, 'obstacle_dilation': 3},
        {'exploration': True, 'use_known_obstacles_in_tv': False, 'use_unknown_obstacles_in_tv': False, 'obstacle_dilation': 5},
        {'exploration': True, 'use_known_obstacles_in_tv': False, 'use_unknown_obstacles_in_tv': True, 'obstacle_dilation': 0}]
    mowers = [
        RandomMower,
        RandomMower,
        RandomMower,
        SpiralMower,
        RandomMower,
        SpiralMower]
    for kwarg, mower in zip(kwargs, mowers):
        env = MowerEnv(**kwarg)
        model = mower(env)
        obs = env.reset()
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done:
                break
        assert env.use_known_obstacles_in_tv == kwarg['use_known_obstacles_in_tv']
        assert env.use_unknown_obstacles_in_tv == kwarg['use_unknown_obstacles_in_tv']
        assert env.obstacle_dilation == kwarg['obstacle_dilation']
        obstacles = None
        if env.use_known_obstacles_in_tv and env.use_unknown_obstacles_in_tv:
            obstacles = np.maximum(env.known_obstacle_map, env.unknown_obstacle_map)
        elif env.use_known_obstacles_in_tv:
            obstacles = env.known_obstacle_map
        elif env.use_unknown_obstacles_in_tv:
            obstacles = env.unknown_obstacle_map
        if obstacles is not None and env.obstacle_dilation > 1:
            kernel = np.ones((env.obstacle_dilation,)*2, dtype=float)
            obstacles[0, :] = 1; obstacles[-1, :] = 1
            obstacles[:, 0] = 1; obstacles[:, -1] = 1
            obstacles = cv2.dilate(obstacles, kernel, iterations=1)
        global_total_variation = utils.total_variation(env.coverage_map, obstacles)
        assert abs(env.global_total_variation - global_total_variation) < 1e-5
        env.close()

def test_action_observation():
    """
    Test for using previous actions as observation.
    """
    steps = 20
    kwargs = [
        {'action_observations': 5, 'constant_lin_vel': True},
        {'action_observations': 10, 'constant_lin_vel': False}]
    mowers = [SpiralMower, SpiralMower]
    for kwarg, mower in zip(kwargs, mowers):
        env = MowerEnv(**kwarg)
        model = mower(env)
        obs = env.reset()
        actions = [[0] * env.action_space.shape[0]] * kwarg['action_observations']
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            actions.append(action)
            oracle = np.array(actions[-kwarg['action_observations']:], dtype=np.float32)
            assert (obs['action'] == oracle).all()
            if done:
                break
        env.close()

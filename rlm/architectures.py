import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class StackedMapFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            features_dim,
            map_size,
            num_maps,
            lidar_rays,
            stacks,
            grouped_convs,
            frontier_observation):
        super(StackedMapFeaturesExtractor, self).__init__(observation_space, features_dim=features_dim)

        num_map_observations = 2
        if frontier_observation:
            num_map_observations = 3

        in_channels = num_map_observations * stacks * num_maps
        out_channels = 2 * in_channels
        out_size = (map_size // 2 - 2 - 2 - 2)**2 * out_channels

        if grouped_convs:
            self.map_extractor = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(out_size, features_dim),
                nn.ReLU()
            )
        else:
            self.map_extractor = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(out_size, features_dim),
                nn.ReLU()
            )

        self.lidar_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(stacks * lidar_rays, lidar_rays),
            nn.ReLU()
        )

        if 'action' in observation_space.keys():
            self.action_extractor = nn.Sequential(
                nn.Flatten()
            )
            action_obs_shape = observation_space['action'].shape
            action_dim = action_obs_shape[0] * action_obs_shape[1]
        else:
            action_dim = 0

        self.fused_extractor = nn.Sequential(
            nn.Linear(features_dim + lidar_rays + action_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> th.Tensor:
        use_frontier = 'frontier' in observations.keys()
        use_action = 'action' in observations.keys()

        # Observations
        lidar = observations['lidar']           # batch x stacks x lidar_rays
        coverage = observations['coverage']     # batch x stacks x nmaps x W x H
        obstacles = observations['obstacles']   # batch x stacks x nmaps x W x H
        if use_frontier:
            frontier = observations['frontier'] # batch x stacks x nmaps x W x H
        if use_action:
            action = observations['action']     # batch x nactions x (1 or 2)

        # Map features
        if use_frontier:
            maps = th.cat([coverage, obstacles, frontier], dim=1) # batch x 3*stacks x nmaps x W x H
        else:
            maps = th.cat([coverage, obstacles], dim=1) # batch x 2*stacks x nmaps x W x H
        maps = maps.permute(0, 2, 1, 3, 4).contiguous() # batch x nmaps x (2 or 3)*stacks x W x H (for grouping correctly)
        b, _, _, w, h = maps.shape
        maps = maps.reshape(b, -1, w, h)                # batch x nmaps*(2 or 3)*stacks x W x H
        map_features = self.map_extractor(maps)         # batch x features_dim

        # Sensor features
        lidar_features = self.lidar_extractor(lidar)    # batch x lidar_rays

        # Action features
        if use_action:
            action_features = self.action_extractor(action) # batch x nactions*(1 or 2)

        # Fused features
        if use_action:
            features = th.cat([map_features, lidar_features, action_features], dim=1)
        else:
            features = th.cat([map_features, lidar_features], dim=1)
        features = self.fused_extractor(features)
        return features

import math
import numpy as np
import random

class RandomMower():
    """
    A mower that performs a random pattern, i.e. drive straight until collision
    and turn a random amount.
    """

    def __init__(self, env):
        self.env = env
        self.num_collisions = 0
        self.is_steering = False
        self.steering_target = 0
        self.steered = 0

    def predict(self, obs, deterministic=None):

        if self.env.num_collisions == 0:
            self.num_collisions = 0
            self.is_steering = False

        if self.env.num_collisions > self.num_collisions:
            self.num_collisions = self.env.num_collisions
            self.is_steering = True
            self.steered = 0
            self.steering_target = random.uniform(-math.pi, math.pi)

        if self.is_steering:
            diff = self.steering_target - self.steered
            max_steering_per_step = self.env.max_ang_vel * self.env.step_size
            if abs(diff) <= max_steering_per_step:
                self.is_steering = False
                if self.env.constant_lin_vel:
                    return [diff / max_steering_per_step], None
                else:
                    return [0, diff / max_steering_per_step], None
            else:
                sgn = np.sign(self.steering_target)
                self.steered += sgn * max_steering_per_step
                if self.env.constant_lin_vel:
                    return [sgn], None
                else:
                    return [0, sgn], None
        
        if self.env.constant_lin_vel:
            return [0], None
        else:
            return [1, 0], None

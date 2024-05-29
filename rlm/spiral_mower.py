import math
import numpy as np
import random

class SpiralMower():
    """
    A mower that performs a spiral pattern, doesn't care abour obstacles.
    """

    def __init__(self, env):
        self.env = env
        self.theta = 0
        self.num_collisions = 0
        self.num_collisions_reset = 10

        self.next_collision_reset = self.num_collisions_reset

    def predict(self, obs, deterministic=None):

        if self.env.elapsed_steps == 0:
            self.theta = 0

        if self.env.num_collisions == 0:
            self.num_collisions = 0
            self.next_collision_reset = self.num_collisions_reset

        if self.num_collisions == self.next_collision_reset:
            self.theta = 0
            self.next_collision_reset += self.num_collisions_reset

        # Spiral parameters
        r0 = 0 # initial radius
        d = 2 * self.env.mower_radius # distance between loops (positive=outward, negative=inward, 0=circle)
        v = self.env.max_lin_vel # speed

        # Compute angular velocity
        if abs(d) < 1e-5:
            # Circle
            w = v / (np.abs(r0) + 1e-5)
        elif abs(v) < 1e-5:
            # Stand still
            v = 0
            w = 0
        else:
            # Archimedean spiral
            a = r0
            b = d / (2 * np.pi)
            r = a + b * self.theta
            w = v / np.sqrt(b ** 2 + r ** 2)

        # Keep within maximum angular velocity
        if w > self.env.max_ang_vel:
            v = v * self.env.max_ang_vel / w
            w = self.env.max_ang_vel
        steering = w / self.env.max_ang_vel
        throttle = v / self.env.max_lin_vel
        self.theta += w * self.env.step_size

        if self.env.num_collisions > self.num_collisions:
            self.num_collisions = self.env.num_collisions
            if self.env.constant_lin_vel:
                return [1], None
            else:
                return [0, 1], None

        if self.env.constant_lin_vel:
            return [steering], None
        else:
            return [throttle, steering], None

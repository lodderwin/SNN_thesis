from collections import deque

import numpy as np


class LandingEnv3D:
    """
    3D MAV control environment based on the formulation in Li et al, 2020.

    'Visual model-predictive localization for computationally efficient autonomous racing of a 72-g drone'
    Source: http://pure.tudelft.nl/ws/portalfiles/portal/73036228/rob.21956.pdf

    Yaw control (psi) has been taken out!
    Body frame: right-handed, Z upwards, X aligned with drone body (not with any arm)
    World frame: right-handed, Z upwards
    """

    # Constants
    G = 9.81

    def __init__(
        self,
        obs_setpoint=[0, 0, 0.5],
        obs_weight=[1.0] * 3,
        obs_delay=0,
        obs_bias=[0.0] * 3,
        obs_noise_std=0.0,
        obs_noise_p_std=0.0,
        wind_std=0.0,
        state_0=[0, 0, 3, 0, 0, 0, 0, 0, 0, G],
        gains=[6, 6, 3.0],
        dt=0.02,
        seed=None,
        act_high=[1 * np.pi] * 2 + [0.4 * G],  #why?
        state_bounds=[[-50, -50, 0.1], [50, 50, 100]],
        time_bound=30,
    ):
        # Parameter dicts:
        # - public for randomizable parameters
        # - protected for non-randomizable ones
        self.param = {
            "obs setpoint": np.array(obs_setpoint),
            "obs weight": np.array(obs_weight),
            "obs delay": obs_delay,
            "obs bias": obs_bias,
            "obs noise std": obs_noise_std,
            "obs noise p std": obs_noise_p_std,
            "wind std": wind_std,
            "state 0": np.array(state_0).reshape(-1, 1),
            "gains": np.array(gains).reshape(-1, 1),
            "dt": dt,
            "seed": seed,
        }
        self._param = {
            "act high": np.array(act_high),
            "state bounds": np.array(state_bounds),
            "time bound": time_bound,
        }

        # Create local RNG
        self.rng = np.random.default_rng(seed)

        # Do checks on arguments
        self._checks()

        # Reset to get initial observation and allocate arrays
        self.reset()

    def _checks(self):
        assert self.param["obs setpoint"].shape[0] == 3
        assert self.param["obs delay"] >= 0 and isinstance(self.param["obs delay"], int)
        assert (
            self.param["obs noise std"] >= 0.0 and self.param["obs noise p std"] >= 0.0
        )
        assert self.param["wind std"] >= 0.0
        assert self.param["state 0"][2, 0] >= 0.0
        assert (self.param["gains"] >= 0.0).all()
        assert self.param["dt"] > 0.0
        assert (self._param["act high"] <= np.array([2 * np.pi] * 2 + [self.G])).all()
        assert (self._param["state bounds"][1] > self._param["state bounds"][0]).all()
        assert self._param["time bound"] > 0.0

    def reset(self):
        # Initial state
        self.state = self.param["state 0"].copy() 
        # starting state: x, y, z, vx, vy, vz, phi, theta, psi, thrust
        self.wind = np.zeros((3, 1))  # wind in three directions
        self.act = np.zeros((3, 1))  #roll pitch thrust

        # Circular array for dealing with delayed observations
        self.obs = deque(maxlen=self.param["obs delay"] + 1)
        self.obs_gt = np.zeros(3)   #true obs with noise

        # Counting variables
        self.done = False
        self.steps = 0
        self.t = 0.0

        # Fill up observations
        for _ in range(self.param["obs delay"] + 1):
            self._get_obs()

        return self.obs[0]

    def set_param(self, param):
        # Check for keys
        assert set(self.param.keys()) >= set(param.keys())

        # Update and reseed (create new RNG again)
        self.param.update(param)
        self._checks()
        self.rng = np.random.default_rng(self.param["seed"])

    def step(self, act):
        # Check if already done last time
        if self.done:
            return self.obs[self.steps], self.reward, self.done

        # Input act is in (-1, 1), scaled with 'act high'
        # Offset G later!
        self.act = (act.clip(-1.0, 1.0) * self._param["act high"]).reshape(-1, 1)

        # Action was taken based on previous observation/state, so now increment step
        self.steps += 1
        self.t += self.param["dt"]

        # Update state with forward Euler
        self.state += self._get_state_dot() * self.param["dt"]

        # Check whether done
        self.done = self._check_out_of_bounds() | self._check_out_of_time()

        # Clamp altitude to bounds (VERY important for reward because of 1/h)
        # TODO: make this nicer?
        if self.done:
            pos_min, pos_max = self._param["state bounds"]
            self.state[0:3] = np.clip(
                self.state[0:3], pos_min.reshape(-1, 1), pos_max.reshape(-1, 1)
            )
            self.t = np.clip(self.t, 0.0, self._param["time bound"])

        # Get reward
        self.reward = self._get_reward()

        return self._get_obs(), self.reward, self.done

    def _get_state_dot(self):
        """
        State: [x, y, z, vx, vy, vz, phi, theta, psi, thrust]^T
        Actions: [phi^c, theta^c, psi^c = 0, thrust^c]^T
        """
        # Position
        # Add wind as velocity disturbance in world frame

        # don't pay attention to the dot label
        p_dot = self.state[3:6] + self._get_wind()
        # Velocity
        v_dot = (
            np.array([0, 0, -self.G]).reshape(-1, 1)
            + self._body2world() @ np.array([0, 0, self.state[9, 0]]).reshape(-1, 1)
            + self._body2world()
            @ np.diagflat([-0.5, -0.5, 0])
            @ self._world2body()
            @ self.state[3:6]
        )
        # Actions
        act = np.insert(self.act, 2, 0, axis=0)  # insert 0 psi^c  # for hovering, so change
        act[3, 0] += self.G  # offset G
        gains = np.insert(self.param["gains"], 2, 0, axis=0)  # and 0 gain
        act_dot = gains * (act - self.state[6:10])

        return np.vstack((p_dot, v_dot, act_dot))

    def _get_wind(self):
        """
        Wind that is correlated over time.
        """
        if self.param["wind std"] > 0.0:
            self.wind += (
                (
                    self.rng.normal(0.0, self.param["wind std"], size=self.wind.shape)
                    - self.wind
                )
                * self.param["dt"]
                / (self.param["dt"] + self.param["wind std"])
            )
        return self.wind

    def _get_obs(self):
        """
        Observation of ventral flows and divergence (visual observables).
        """
        # Compute observation
        self.obs_gt[0] = -self.state[3, 0] / np.maximum(1e-5, self.state[2, 0])  # /z !?
        self.obs_gt[1] = -self.state[4, 0] / np.maximum(1e-5, self.state[2, 0])
        self.obs_gt[2] = -2 * self.state[5, 0] / np.maximum(1e-5, self.state[2, 0])


        # Add bias + noise
        # TODO: can this be done vectorized?
        wx = (
            self.obs_gt[0]
            + self.rng.normal(self.param["obs bias"][0], self.param["obs noise std"])
            + abs(self.obs_gt[0]) * self.rng.normal(0.0, self.param["obs noise p std"])
        )
        wy = (
            self.obs_gt[1]
            + self.rng.normal(self.param["obs bias"][1], self.param["obs noise std"])
            + abs(self.obs_gt[1]) * self.rng.normal(0.0, self.param["obs noise p std"])
        )
        div = (
            self.obs_gt[2]
            + self.rng.normal(self.param["obs bias"][2], self.param["obs noise std"])
            + abs(self.obs_gt[2]) * self.rng.normal(0.0, self.param["obs noise p std"])
        )

        # Append to end of deque
        self.obs.append(np.array([wx, wy, div]))

        return self.obs[0]

    def _get_reward(self):
        return (
            1
            - (
                np.abs(self.obs_gt - self.param["obs setpoint"])
                * self.param["obs weight"]
            ).sum()
        ).clip(-1.0, 1.0)

    def _body2world(self):
        return body2world(*self.state[6:9, 0])

    def _world2body(self):
        return world2body(*self.state[6:9, 0])

    def _check_out_of_bounds(self):
        pos_min, pos_max = self._param["state bounds"]
        return (
            (self.state[0:3] <= pos_min.reshape(-1, 1))
            | (self.state[0:3] >= pos_max.reshape(-1, 1))
        ).any()

    def _check_out_of_time(self):
        return (self.steps + 1) * self.param["dt"] >= self._param["time bound"]


def body2world(phi, theta, psi):
    """
    Rotation matrix from body frame to world frame.
    """
    c, s = np.cos, np.sin
    rot = np.array(
        [
            [
                c(psi) * c(theta),
                c(psi) * s(theta) * s(phi) - s(psi) * c(phi),
                c(psi) * s(theta) * c(phi) + s(psi) * s(phi),
            ],
            [
                s(psi) * c(theta),
                s(psi) * s(theta) * s(phi) + c(psi) * c(phi),
                s(psi) * s(theta) * c(phi) - c(psi) * s(phi),
            ],
            [-s(theta), c(theta) * s(phi), c(theta) * c(phi)],
        ]
    )
    return rot


def world2body(phi, theta, psi):
    """
    Rotation matrix from world frame to body frame.
    """
    return body2world(phi, theta, psi).T
